import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        #x: (bs,tokens,channels) or (bs,channels,tokens)
        self.fc1.to(x.device)
        self.fc2.to(x.device)
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=16, max_len=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super(MixerBlock, self).__init__()
        self.ln = nn.LayerNorm(max_len)
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(max_len, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        # tokens mixing
        self.ln.to(x.device).to(torch.float32)
        y = self.ln(x.float())
        y = y.transpose(1, 2)  # (bs,channels,tokens)
        y = self.tokens_mlp_block(y)  # (bs,channels,tokens)
        # channels mixing
        y = y.transpose(1, 2)  # (bs,tokens,channels)
        out = x + y  # (bs,tokens,channels)
        y = self.ln(out.float())  # (bs,tokens,channels)
        y = out + self.channels_mlp_block(y)  # (bs,tokens,channels)
        return y


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim, channels_hidden_dim, tokens_mlp_dim,
                 max_len):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks  # num of mlp layers
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.max_len = max_len
        self.embd = nn.Conv2d(1, max_len, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(max_len)
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim, max_len, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Linear(max_len, num_classes)

    def forward(self, y):
        # y = y.unsqueeze(1)

        if (self.tokens_mlp_dim != y.shape[1]):
            raise ValueError('Tokens_mlp_dim is not correct.')

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)  # bs,tokens,channels
        y = self.ln(y.float())  # bs,tokens,channels
        y = torch.mean(y, dim=1, keepdim=False)  # bs,channels
        probs = self.fc(y)  # bs,num_classes
        return probs


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


class GradMixer(nn.Module):
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
        super(GradMixer, self).__init__()

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
        self.Sit = MlpMixer(num_classes=num_class,
                            num_blocks=4,
                            patch_size=6,
                            tokens_hidden_dim=128,
                            channels_hidden_dim=mlp_dim,
                            tokens_mlp_dim=self.signal_heigh,
                            max_len=self.signal_width)

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight.data.fill_(0.02)  # = 初始化
        self.sm = nn.Tanh()
        self.latent = nn.Identity()

        self.embed = nn.Sequential(nn.Linear(max_len, self.signal_width), nn.Dropout(0.1))

    def forward(self, x):
        if not self.raw_embed:
            x = self.gradConv(x)
            x = self.Sit(x)
        else:
            raw = self.latent(x)
            raw = self.embed(raw) * self.sm(self.weight)

            x = self.gradConv(x)

            x = torch.cat((raw.unsqueeze(1), x), dim=1)
            x = self.Sit(x)

        return x


if __name__ == "__main__":
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
                      raw_embed=True).cuda()
    img = torch.rand(1, 501).cuda()
    from magnet_models2D import FPS
    FPS(model, (img, ))
    res = model(img)

    print(res.shape)
