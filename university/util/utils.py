import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class EarlyStopping():
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'{Fore.RED}EarlyStopping counter: {self.counter} out of {self.patience}{Style.RESET_ALL}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'{Fore.GREEN}Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...{Style.RESET_ALL}'
            )
        # torch.save(model.state_dict(), path + '_checkpoint.pth')
        self.val_loss_min = val_loss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class argparse():
    pass


def metric(outputs, label):
    """[计算检测率和虚警值]

    Args:
        outputs ([type]): [神经网络输出]
        label ([type]): [ground truth]

    Returns:
        [P, N, detect, false_alarm]: [阳性，阴性，检测值，虚警值]
        [warning]: 是个数不是百分比
    """
    out = outputs.max(axis=1)[1]
    P = label.sum().item()
    detect = (out & label).sum().item()
    false_alarm = out.sum().item() - detect
    N = label.size()[0] - P

    return P, N, detect, false_alarm
    """[usage]
        P, N, detect, false_alarm = 0, 0, 0, 0
            elem1, elem2, elem3, elem4 = metric(outputs=outputs, label=label)
            P += elem1
            N += elem2
            detect += elem3
            false_alarm += elem4
        print("detect = {:.3f}%, miss_rate = {:.3f}%, false_alarm = {:.3f}%".format(
            100*detect/P, 100.-100*detect/P, 100*false_alarm/N))
    """
    """_summary_

    Returns:
        _type_: _description_
    """


def metric2(outputs, label):
    """[计算检测率和虚警值]

    Args:
        outputs ([type]): [神经网络输出]
        label ([type]): [ground truth]

    Returns:
        [P, N, detect, false_alarm]: [阳性，阴性，检测值，虚警值]
        [warning]: 是个数不是百分比
    """
    out = outputs.ge(0.8).max(axis=1)[1]
    P = label.sum().item()
    detect = (out & label).sum().item()
    false_alarm = out.sum().item() - detect
    N = label.size()[0] - P

    return P, N, detect, false_alarm
    """[usage]
        P, N, detect, false_alarm = 0, 0, 0, 0
            elem1, elem2, elem3, elem4 = metric(outputs=outputs, label=label)
            P += elem1
            N += elem2
            detect += elem3
            false_alarm += elem4
        print("detect = {:.3f}%, miss_rate = {:.3f}%, false_alarm = {:.3f}%".format(
            100*detect/P, 100.-100*detect/P, 100*false_alarm/N))
    """
    """_summary_

    Returns:
        _type_: _description_
    """


class Dataset_signal_textcnn2(Dataset):
    def __init__(self, flag='train', scale=True, shift=20, second_shift_depart=5) -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        self.shift = shift
        self.second_shift_depart = second_shift_depart
        ann = pd.read_csv("/home/wfnian/signal/workspace/data/C2SNR_6_6/{}_ann.txt".format(self.flag),
                          sep=', ',
                          engine='python',
                          header=None).values
        # ann = pd.read_csv("/home/wfnian/signal/workspace/data/sin_cos.txt".format(self.flag),
        #                   sep=', ', engine='python', header=None).values

        self.data = ann

        if scale:

            self.scaler = StandardScaler()
            self.scaler.fit(self.data[:, 2:])
            self.data = np.concatenate((self.data[:, :2], self.scaler.transform(self.data[:, 2:])), axis=1)

    def __getitem__(self, index: int):
        val = self.data[index]
        label = int(val[0])
        seq = val[2:]
        pe = torch.zeros(self.shift + self.shift * self.shift // self.second_shift_depart,
                         501 - self.shift - self.shift // self.second_shift_depart)
        idx_down, idx_up = 0, -1
        for sf in range(self.shift):
            temp = ((seq[sf + 1:] - seq[:-sf - 1])[-(501 - self.shift):])
            pe[idx_down] = torch.tensor(temp[-(len(temp) - self.shift // self.second_shift_depart):])
            idx_down = idx_down + 1
            for deriv2 in range(self.shift // self.second_shift_depart):
                pe[idx_up] = torch.tensor(
                    ((temp[deriv2 + 1:] -
                      temp[:-deriv2 - 1])[-(len(temp) - self.shift // self.second_shift_depart):]))
                idx_up -= 1

        pe = pe.unsqueeze(0)

        return torch.tensor(label, dtype=torch.long), pe

    def __len__(self) -> int:
        print("=" * 100)
        return len(self.data)


class Dataset_signal_vit_2C(Dataset):
    def __init__(self, flag='train', scale=True) -> None:
        assert flag in ['train', 'val'], 'not implement!'
        self.flag = flag
        ann = pd.read_csv("/home/wfnian/signal/workspace/data/C2SNR_2_6/{}_ann.txt".format(self.flag),
                          sep=',',
                          header=None).values

        self.data = ann

        if scale:

            self.scaler = StandardScaler()
            self.scaler.fit(self.data[:, 2:])
            self.data = np.concatenate((self.data[:, :2], self.scaler.transform(self.data[:, 2:])), axis=1)

    def __getitem__(self, index: int):
        val = self.data[index]
        label = int(val[0])
        seq = val[2:]
        # seq = (seq-MIN_VAL)/(MAX_VAL-MIN_VAL)*1000
        seq = seq[1:] - seq[:-1]
        seq = seq * 1000
        pad = np.zeros(88, )
        seq = np.concatenate((seq, pad)).reshape(3, 14, 14)
        return torch.tensor(label, dtype=torch.long), torch.tensor(seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss2, self).__init__()
        self._gamma = gamma
        self._alpha = alpha

    def forward(self, y_pred, y_true):
        cross_entropy_loss = torch.nn.BCELoss(y_true, y_pred)
        p_t = ((y_true * y_pred) + ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha + (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * cross_entropy_loss)
        return focal_cross_entropy_loss.mean()


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1) + 1e-6
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
        return self.linear_combination(loss / n, nll)


def plotfigure(train_acc, val_acc, train_epochs_loss, valid_epochs_loss, setting, bestScore, wechat=False):
    import matplotlib.pyplot as plt
    import requests

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_acc[:], '-', label="train_acc")
    plt.plot(val_acc[:], '-', label="acc@1")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-', label="train_loss")
    plt.plot(valid_epochs_loss[1:], '-', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.grid()
    plt.savefig(setting + str(bestScore) + ".png")
    # plt.show()

    if wechat:
        requests.get(
            "http://www.pushplus.plus/send?token=24af4bfe58114caebafb91a10cf2f1df&title=程序通知&content={}&template=html"
            .format(bestScore))


def inform(bestScore, wechat=False):
    import requests
    if wechat:
        requests.get(
            "http://www.pushplus.plus/send?token=24af4bfe58114caebafb91a10cf2f1df&title=程序通知&content={}&template=txt"
            .format(bestScore))


def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s, ' f'times per image: {1000 / fps:.1f} ms / img', flush=True)
            break
    return fps
