import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


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


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    groups=self.groups)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups,
                 downsample,
                 use_bn,
                 use_do,
                 is_first_block=False):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=self.stride,
                                     groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """
    def __init__(self,
                 in_channels,
                 base_filters,
                 kernel_size,
                 stride,
                 groups,
                 n_block,
                 n_classes,
                 downsample_gap=2,
                 increasefilter_gap=4,
                 use_bn=True,
                 use_do=True,
                 verbose=False,
                 shift_grad=True):
        super(ResNet1D, self).__init__()

        self.shift_grad = shift_grad

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels,
                                                out_channels=base_filters,
                                                kernel_size=self.kernel_size,
                                                stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2**((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   groups=self.groups,
                                   downsample=downsample,
                                   use_bn=self.use_bn,
                                   use_do=self.use_do,
                                   is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        if not self.shift_grad:
            x = x.unsqueeze(1)

        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(
                    i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)

        return out


class GradResNet1D(nn.Module):
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
        super(GradResNet1D, self).__init__()

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

        # self.textCNN = TextCNN_v0(embedding_dim=self.signal_heigh, max_len=self.signal_width, num_classes=num_class)
        self.resnet1d = ResNet1D(in_channels=self.signal_width,
                                 base_filters=64,
                                 kernel_size=3,
                                 stride=3,
                                 groups=8,
                                 n_block=18,
                                 n_classes=num_class)
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
        x = x.permute(0, 2, 1)
        x = self.resnet1d(x)
        return x


if __name__ == "__main__":
    model1 = GradResNet1D(max_len=501,
                          num_class=2,
                          poolKernel=1,
                          shift=2,
                          isDerive2=False,
                          dropout=0,
                          emb_dropout=0,
                          shift2=0,
                          batchNorm=True,
                          layerNorm=True,
                          isMs=False,
                          raw_embed=False).cuda()
    model = ResNet1D(in_channels=1,
                     base_filters=64,
                     kernel_size=3,
                     stride=3,
                     groups=8,
                     n_block=18,
                     n_classes=2,
                     shift_grad=False).cuda()
    img = torch.rand(1, 501).cuda()
    from magnet_models2D import FPS
    FPS(model1, (img, ))
    # res = model(img)

    # print(res.shape)
