import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore, Style
from torch.nn import utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from thop import profile

from dataloaders import (esc50Dataloader, indianDataloader, magnetDataloader, paviaDataloader, polsarDataloader,
                         salinasDataloader, paviaDisDataloader, indianDisDataloader, ecgDataloader, ecgDataloader2d)
from model.GradSit import GradSit
from models.GradVit import GradVit
from models.TextCNN import TextCNN_v0
from models.GradTextCNN import GradTextCNN
from models.ResNet1D import GradResNet1D, ResNet1D
from models.SimpleCNN1D import SimpleCNN1D
from models.SimpleCNN1D2 import SimpleCNN1D1
from models.GradMixer import GradMixer
from models.Mixer import MlpMixer
# from models.ecg_models1D import ZolotyhNet, HeartNetIEEE, HeartNet, EcgResNet34, Anomaly_Classifier
from models.ecg_models2D import HeartNet, EfficientNetB4, ResNet34
from util.utils import (AverageMeter, EarlyStopping, FocalLoss, LabelSmoothingLoss, StandardScaler, accuracy,
                        argparse, metric, plotfigure, inform)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


class Args():
    def __init__(self) -> None:
        self.epochs = 800
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.head = 4
        self.depth = 4
        self.dim = 512  #~512 0.2 最佳
        self.isDerive2 = True
        self.shift = 2
        self.shift2 = 9
        self.learning_rate = 0.001
        self.pool = 3
        self.batchNorm = True
        self.layerNorm = False
        self.batch_size = 1536
        self.dropout = 0.0  #~ 0.2目前最佳
        self.patience = 30
        self.is_ms = True
        self.raw_embed = True
        self.bestScore = 0.
        self.recode = [0., 0., 0.]

        #: 更改数据集名称
        self.data = 'magnet2'
        #: 无需修改的参数
        self.max_len = None
        self.num_class = None


def main():
    args = Args()

    data_type = {
        'india': {
            'max_len': 220,
            'nums_class': 16,
            'function': indianDataloader.getDataLoader
        },
        'indiaDis': {
            'max_len': 220,
            'nums_class': 16,
            'function': indianDisDataloader.getDataLoader
        },
        'pavia': {
            'max_len': 103,
            'nums_class': 9,
            'function': paviaDataloader.getDataLoader
        },
        'paviaDis': {
            'max_len': 103,
            'nums_class': 9,
            'function': paviaDisDataloader.getDataLoader
        },
        'salinas': {
            'max_len': 204,
            'nums_class': 7,
            'function': salinasDataloader.getDataLoader
        },
        'polsar': {
            'max_len': 109,
            'nums_class': 11,
            'function': polsarDataloader.getDataLoader
        },
        'magnet2': {
            'max_len': 501,
            'nums_class': 2,
            'function': magnetDataloader.getDataLoader  # magnet2.getDataLoader
        },
        'magnet256': {
            'max_len': 501,
            'nums_class': 256,
            'function': NotImplemented  # magnet2.getDataLoader
        },
        'esc50': {  # audio 50
            'max_len': 5000,
            'nums_class': 50,
            'function': esc50Dataloader.getDataLoader
        },
        'ecg': {  # audio 50
            'max_len': 128,
            'nums_class': 8,
            'function': ecgDataloader2d.getDataLoader
            # 'function': ecgDataloader.getDataLoader
        }
    }
    args.max_len = data_type[args.data]['max_len']
    args.num_class = data_type[args.data]['nums_class']
    getDataLoader = data_type[args.data]['function']

    train_dataloader, valid_dataloader = getDataLoader(batch_size=args.batch_size)
    # model = GradSit(heads=args.head,
    #                 isDerive2=args.isDerive2,
    #                 shift=args.shift,
    #                 poolKernel=args.pool,
    #                 shift2=args.shift2,
    #                 batchNorm=args.batchNorm,
    #                 layerNorm=args.layerNorm,
    #                 num_class=args.num_class,
    #                 max_len=args.max_len,
    #                 dropout=args.dropout,
    #                 emb_dropout=args.dropout,
    #                 dim=args.dim,
    #                 mlp_dim=args.dim,
    #                 isMs=args.is_ms,
    #                 raw_embed=args.raw_embed).to(args.device).to(torch.float32)
    # model = MlpMixer(num_classes=2,
    #                  num_blocks=4,
    #                  patch_size=6,
    #                  tokens_hidden_dim=64,
    #                  channels_hidden_dim=512,
    #                  tokens_mlp_dim=1,
    #                  max_len=501).cuda().to(torch.float32)
    model = GradMixer(max_len=501,
                      num_class=2,
                      poolKernel=1,
                      shift=2,
                      isDerive2=False,
                      dropout=0,
                      emb_dropout=0,
                      shift2=0,
                      batchNorm=True,
                      layerNorm=True,
                      isMs=True,
                      raw_embed=True).cuda().to(torch.float32)
    # model = TextCNN_v0(max_len=args.max_len, num_classes=args.num_class).to(args.device).to(torch.float32)
    #region
    # model.load_state_dict(
    #     torch.load(
    #         r'/home/wfnian/heart/universality/figs/datapaviaDisLWDB_head4_depth4_dim512_drFalse_sf3_sf21_lr0.001_pool9_bnTrue_lnFalse_bz512_dpot0.1_pt80_msFalse_rawTrue_mdGradSitGradSit_checkpoint22-84.pth'
    #     ))
    # model = GradTextCNN(max_len=args.max_len,
    #                     num_class=args.num_class,
    #                     poolKernel=args.pool,
    #                     shift=args.shift,
    #                     isDerive2=args.isDerive2,
    #                     dropout=args.dropout,
    #                     emb_dropout=args.dropout,
    #                     shift2=args.shift2,
    #                     batchNorm=args.batchNorm,
    #                     layerNorm=args.layerNorm,
    #                     isMs=args.is_ms,
    #                     raw_embed=True).to(args.device).to(torch.float32)
    # model = GradResNet1D(max_len=args.max_len,
    #                      num_class=args.num_class,
    #                      poolKernel=args.pool,
    #                      shift=args.shift,
    #                      isDerive2=args.isDerive2,
    #                      dropout=args.dropout,
    #                      emb_dropout=args.dropout,
    #                      shift2=args.shift2,
    #                      batchNorm=args.batchNorm,
    #                      layerNorm=args.layerNorm,
    #                      isMs=args.is_ms,
    #                      raw_embed=False).to(args.device).to(torch.float32)
    # model = GradVit(max_len=args.max_len,
    #                 num_class=args.num_class,
    #                 poolKernel=args.pool,
    #                 shift=args.shift,
    #                 isDerive2=args.isDerive2,
    #                 dropout=args.dropout,
    #                 emb_dropout=args.dropout,
    #                 shift2=args.shift2,
    #                 batchNorm=args.batchNorm,
    #                 layerNorm=args.layerNorm,
    #                 isMs=args.is_ms,
    #                 raw_embed=False).to(args.device).to(torch.float32)
    # model = ResNet1D(in_channels=1,
    #                  base_filters=32,
    #                  kernel_size=3,
    #                  stride=3,
    #                  groups=8,
    #                  n_block=12,
    #                  n_classes=args.num_class,
    #                  shift_grad=False).to(args.device).to(torch.float32)
    # model = SimpleCNN1D(args.max_len, args.num_class, pool=False, n_cnn_kernel=3).to(args.device)
    # model = SimpleCNN1D1(input_channels=args.max_len,
    #                      n_kernels=20,
    #                      kernel_size=10,
    #                      pool_size=5,
    #                      n4=100,
    #                      n_classes=args.num_class).to(args.device)
    #endregion


    setting = "/home/wfnian/heart/universality/figs/" + \
            "data{}_head{}_depth{}_dim{}_dr{}_sf{}_sf2{}_lr{}_pool{}_bn{}_ln{}_bz{}_dpot{}_pt{}_ms{}_raw{}_md{}".format(
            args.data, args.head, args.depth,  args.dim, args.isDerive2, args.shift, args.shift2,
            args.learning_rate, args.pool, args.batchNorm,args.layerNorm, args.batch_size,
            args.dropout, args.patience,args.is_ms,args.raw_embed,model._get_name())
    setting2 = "data_{}_model_{}".format(args.data, model._get_name())
    criterion = nn.CrossEntropyLoss()

    # criterion = FocalLoss(args.num_class, gamma=2)
    criterion = LabelSmoothingLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.02,
                                dampening=0.618)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  #, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=2e-7)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                                                  T_0=6,
    #                                                                  T_mult=2,
    #                                                                  eta_min=2e-7,
    #                                                                  last_epoch=-1)
    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []
    early_stopping = EarlyStopping(verbose=True, patience=args.patience)

    for epoch in range(args.epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        for idx, (label, signal) in enumerate(tqdm(train_dataloader)):
            signal = signal.to(args.device)
            label = label.to(args.device)
            outputs = model(signal)
            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += sum(outputs.max(axis=1)[1] == label).cpu()
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100 * acc / nums)
        print("train acc = {:.3f}%,loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        torch.cuda.empty_cache()
        # =========================val=========================
        with torch.no_grad():
            model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0
            P, N, detect, false_alarm = 0, 0, 0, 0
            gt, pd = [], []
            for idx, (label, signal) in enumerate(tqdm(valid_dataloader)):
                signal = signal.to(args.device)  #.to(torch.float)
                label = label.to(args.device)
                outputs = model(signal)
                loss = criterion(outputs, label)
                val_epoch_loss.append(loss.item())

                acc += sum(outputs.max(axis=1)[1] == label).cpu()
                nums += label.size()[0]

                elem1, elem2, elem3, elem4 = metric(outputs=outputs, label=label)
                P += elem1
                N += elem2
                detect += elem3
                false_alarm += elem4
                gt += label.tolist()
                pd += (outputs.max(axis=1)[1]).tolist()

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(100 * acc / nums)

            if val_acc[-1] > args.bestScore:
                args.bestScore = val_acc[-1]
                args.recode = [100 * detect / P, 100. - 100 * detect / P, 100 * false_alarm / N]
                Gt, Pd = gt, pd

            print("epoch = {}, valid acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums,
                                                                      np.average(val_epoch_loss)))

            print("detect = {:.2f}%, miss_rate = {:.2f}%, false_alarm = {:.2f}%".format(
                100 * detect / P, 100. - 100 * detect / P, 100 * false_alarm / N))

        # ==================early stopping=====================
        early_stopping(valid_epochs_loss[-1], model=model, path=setting + model._get_name())
        if early_stopping.early_stop:
            np.save('/home/wfnian/heart/universality/res/{}_{}_pred.npy'.format(setting2, args.bestScore),
                    np.array(Pd))
            np.save('/home/wfnian/heart/universality/res/{}_{}_gt.npy'.format(setting2, args.bestScore),
                    np.array(Gt))
            print("early_stopping!!!")
            inform(bestScore=args.bestScore, wechat=True)
            break
        # ==================adjust lr==========================
        # scheduler.step()
        if early_stopping.counter in [8, 16, 32, 53, 128, 192, 256]:
            # if early_stopping.counter in [60, 128, 192, 256]:
            lr = args.learning_rate / 2
            args.learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('{}Updating learning rate to = {}{}'.format(Fore.BLUE, lr, Style.RESET_ALL))
        # print('{}Updating learning rate to = {}{}'.format(Fore.BLUE, scheduler.get_last_lr(), Style.RESET_ALL))
        # print("=" * 36 + "lr = {} bestScore = {:.3f}".format(args.learning_rate, args.bestScore))
        print("=" * 36 + "lr = {} bestScore = {:.3f} another = {}".format(
            args.learning_rate, args.bestScore, list(map(lambda fc: round(fc, 2), args.recode))))

    plotfigure(train_acc, val_acc, train_epochs_loss, valid_epochs_loss, setting, args.bestScore, wechat=False)


if __name__ == "__main__":
    main()
