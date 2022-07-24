import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
                       if project_out else nn.Identity())

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SiT(nn.Module):
    def __init__(self,
                 *,
                 signal_heigh,
                 signal_width,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool="cls",
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0):
        super().__init__()

        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(nn.Linear(signal_width, dim), )

        self.pos_embedding = nn.Parameter(torch.randn(1, signal_heigh + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


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
        # self.weight.data.fill_(0.618)  # = 初始化
        self.sm = nn.Hardtanh()

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
            res1 = res1 * self.sm(self.weight)
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


class GradSit(nn.Module):
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
        super(GradSit, self).__init__()

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
        self.Sit = SiT(
            signal_heigh=self.signal_heigh,
            signal_width=self.signal_width,
            num_classes=num_class,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.weight.data.fill_(0.02)  # = 初始化 Pavia0.02 Paviadis 0.99
        self.sm = nn.Hardtanh()
        # self.sm = nn.Tanh()
        self.latent = nn.Identity()

        self.embed = nn.Sequential(nn.Linear(max_len, self.signal_width), nn.Dropout(0.1))

    #     self.init_weight()

    # def init_weight(self):
    #     for m in self.modules():
    #         # if isinstance(m, nn.BatchNorm2d):
    #         #     nn.init.constant_(m.weight, 1)
    #         #     nn.init.constant_(m.bias, 0)
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform(m.weight)

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
    model = GradSit(max_len=200,
                    num_class=15,
                    poolKernel=1,
                    shift=2,
                    isDerive2=False,
                    dropout=0,
                    emb_dropout=0,
                    shift2=0,
                    batchNorm=True,
                    layerNorm=True,
                    isMs=True,
                    raw_embed=False)
    img = torch.rand(4, 200)
    res = model(img)

    print(res.shape)
