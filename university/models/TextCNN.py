import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class TextCNN_v0(nn.Module):
    def __init__(self, *, embedding_dim=1, max_len, num_classes):
        super(TextCNN_v0, self).__init__()
        self.num_cls = num_classes
        self.conv3 = nn.Conv2d(1, 1, (3, embedding_dim))
        self.conv4 = nn.Conv2d(1, 1, (4, embedding_dim))
        self.conv5 = nn.Conv2d(1, 1, (5, embedding_dim))
        self.Max4_pool = nn.MaxPool2d((max_len - 4 + 1, 1))
        self.Max3_pool = nn.MaxPool2d((max_len - 3 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((max_len - 5 + 1, 1))
        self.linear1 = nn.Linear(3, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, self.num_cls)

        return x


class TextCNN_v1(nn.Module):
    def __init__(
            self,
            embedding_dim=1,  # embed dim
            num_classes=2,
            filter_num=2,
            max_len=198,
            kernel_list=(3, 5, 7, 9),
            dropout=0.5):
        super(TextCNN_v1, self).__init__()
        assert sum(kernel_list) % 2 == 0, " sum(kernel_list) must divided by 2"

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)), nn.ReLU(), nn.MaxPool2d((kernel, 1)))
            for kernel in kernel_list
        ])
        # print(sum([(filter_num * ((500 - kernel + 1 - kernel) // kernel + 1)) for kernel in kernel_list]))
        self.fc = nn.Linear(
            sum([(filter_num * ((max_len - kernel + 1 - kernel) // kernel + 1)) for kernel in kernel_list]),
            num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(x.shape)
        x = x.unsqueeze(1)
        out = [conv(x).view(1, -1, 1) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)

        logits = self.fc(out)
        return logits


class DeriveLayer(nn.Module):
    def __init__(self,
                 shift=3,
                 isDeriv2=True,
                 shift2=1,
                 poolKernel=9,
                 batchNorm=False,
                 layerNorm=False,
                 max_len: int = None,
                 isMs=True):
        super(DeriveLayer, self).__init__()

        assert shift > 0, "shift must >=1"
        if isDeriv2:
            assert shift2 > 0, "if isDerive2, shift2 must >=1"

        self.derivConv2d = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=(2, 1),
                                     stride=(2, 1),
                                     padding=0,
                                     bias=False)
        self.derivConv2d.apply(self.conv_init)

        self.AvgPool = nn.AvgPool1d(kernel_size=poolKernel, stride=1) if poolKernel > 2 else nn.Identity()
        self.shift = shift
        self.isDeriv2 = isDeriv2
        self.shift2 = shift2
        self.isMs = isMs

        self.latent = nn.Identity()
        for param in self.parameters():
            param.requires_grad = False

        if isDeriv2:
            self.width = (max_len - max(1 << (shift - 1), (1 << (shift2 - 1))) - poolKernel +
                          1) if poolKernel > 2 else (max_len - max(1 << (shift - 1), (1 << (shift2 - 1))))
        else:
            self.width = (max_len - (1 <<
                                     (shift - 1)) - poolKernel + 1) if poolKernel > 2 else (max_len - (1 <<
                                                                                                       (shift - 1)))
        self.bn = nn.BatchNorm2d(1) if batchNorm else nn.Identity()
        self.ln = nn.LayerNorm(self.width) if layerNorm else nn.Identity()

        self.ms = MultiScale(poolKernel, self.width)

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(0.618)  # = 初始化
        self.sm = nn.ReLU()

    def forward(self, x):
        x = x[:, None, None, :]  # ~ bz,1,1,len

        x = x.repeat(1, 1, self.shift << 1, 1)  # ~ bz,1,shift*2,len
        for d1 in range(0, self.shift):
            x[:, :, d1 << 1, :] = x.roll(1 << d1, -1)[:, :, d1 << 1, :]
        deriv1 = self.derivConv2d(x)
        if self.isDeriv2:
            deriv2 = self.latent(deriv1)[:, :, None, 0]
            deriv2 = deriv2.repeat(1, 1, self.shift2 << 1, 1)
            for d2 in range(0, self.shift2):
                deriv2[:, :, d2 << 1, :] = deriv2.roll(1 << d2, -1)[:, :, d2 << 1, :]
            res1 = self.derivConv2d(deriv2)
            #res1 = res1 * self.sm(self.weight)
            res = torch.cat((deriv1, res1), dim=-2)
            res = res[:, :, :, max((1 << d1), (1 << d2)):]
        else:
            res = deriv1[:, :, :, (1 << d1):]

        # TODO higher derivate. note that derive2 used derive1[:,:,:,0]
        # TODO                  derive3 should use derive2[:,:,:,0]

        res = self.bn(res)
        res = res.squeeze(1)
        res = self.AvgPool(res)
        if self.isMs:
            res = self.ms(res)
        else:
            res = res.repeat(1, 4, 1)
        res = self.ln(res)

        return res

    def conv_init(self, conv):
        weight = torch.tensor(
            [[[[-1.0], [1.0]]]],
            requires_grad=False)  # .to(torch.device("cuda:7" if torch.cuda.is_available() else "cpu"))

        conv.weight.data = weight


class MultiScale(nn.Module):
    def __init__(self, poolKernel=9, size: int = None) -> None:
        super(MultiScale, self).__init__()
        poolKernel = 9
        self.avgPool1 = nn.AvgPool1d(kernel_size=poolKernel, stride=1)
        self.avgPool2 = nn.AvgPool1d(kernel_size=poolKernel, )
        self.maxPool1 = nn.MaxPool1d(kernel_size=poolKernel, stride=1)
        self.maxPool2 = nn.MaxPool1d(kernel_size=poolKernel, )
        self.upSample = nn.Upsample(size=(size), mode="nearest")
        self.latent = nn.Identity()

    def forward(self, x):
        res = [
            self.upSample(self.avgPool1(x)) + x,
            self.upSample(self.avgPool2(x)) + x,
            self.upSample(self.maxPool1(x)) + x,
            self.upSample(self.maxPool2(x)) + x,
        ]
        return torch.cat(res, dim=1)


class GradTextCNN(nn.Module):
    def __init__(self,
                 *,
                 max_len,
                 num_class,
                 shift,
                 isDerive2,
                 dim=1024,
                 depth=6,
                 heads=4,
                 mlp_dim=1024,
                 dropout,
                 emb_dropout,
                 poolKernel,
                 shift2,
                 batchNorm,
                 layerNorm,
                 isMs,
                 raw_embed=True):
        super(GradTextCNN, self).__init__()

        self.gradConv = DeriveLayer(shift=shift,
                                    isDeriv2=isDerive2,
                                    shift2=shift2,
                                    poolKernel=poolKernel,
                                    batchNorm=batchNorm,
                                    layerNorm=layerNorm,
                                    max_len=max_len,
                                    isMs=isMs)
        if isDerive2:
            self.signal_heigh = (shift + shift2) * 4 + 1
            self.signal_width = (max_len - max(1 << (shift - 1), (1 << (shift2 - 1))) - poolKernel + 1)
            self.signal_width = (max_len - max(1 << (shift - 1), (1 << (shift2 - 1))) - poolKernel +
                                 1) if poolKernel > 2 else (max_len - max(1 << (shift - 1), (1 << (shift2 - 1))))
        else:
            self.signal_heigh = shift * 4 + 1  # ~ 4 represent ms 4 times,and 1 is embed raw data
            self.signal_width = (max_len - (1 << (shift - 1)) - poolKernel + 1)
            self.signal_width = (max_len - (1 << (shift - 1)) - poolKernel +
                                 1) if poolKernel > 2 else (max_len - (1 << (shift - 1)))
        self.raw_embed = raw_embed
        if not self.raw_embed:
            self.signal_heigh -= 1

        self.textCNN = TextCNN_v0(embedding_dim=self.signal_heigh, max_len=self.signal_width, num_classes=num_class)
        # self.Sit = SiT(
        #     signal_heigh=self.signal_heigh,
        #     signal_width=self.signal_width,
        #     num_classes=num_class,
        #     dim=dim,
        #     depth=depth,
        #     heads=heads,
        #     mlp_dim=mlp_dim,
        #     dropout=dropout,
        #     emb_dropout=emb_dropout,
        # )

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(0.02)  # = 初始化
        self.sm = nn.Tanh()
        self.latent = nn.Identity()

        self.embed = nn.Sequential(nn.Linear(max_len, self.signal_width), nn.Dropout(0.1))

    def forward(self, x):
        if not self.raw_embed:
            x = self.gradConv(x)

        else:
            raw = self.latent(x)
            raw = self.embed(raw) * self.sm(self.weight)

            x = self.gradConv(x)

            x = torch.cat((raw.unsqueeze(1), x), dim=1)

        # print(x.shape)
        # x = x.permute(0, 2, 1)
        x = self.textCNN(x)
        return x


if __name__ == "__main__":
    model = TextCNN_v0(embedding_dim=1, max_len=501, num_classes=2).cuda()
    # model = GradTextCNN(max_len=501,
    #                     num_class=2,
    #                     poolKernel=9,
    #                     shift=3,
    #                     isDerive2=False,
    #                     dropout=0,
    #                     emb_dropout=0,
    #                     shift2=0,
    #                     batchNorm=True,
    #                     layerNorm=False,
    #                     isMs=True,
    #                     raw_embed=True).cuda()
    img = torch.rand(1, 501).cuda()
    res = model(img)
    from magnet_models2D import FPS
    FPS(model, (img, ))

    print(res.shape)
