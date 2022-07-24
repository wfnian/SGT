import torch
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
        return self.fc2(self.gelu(self.fc1(x.double())))


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
        y = y.unsqueeze(1)

        if (self.tokens_mlp_dim != y.shape[1]):
            raise ValueError('Tokens_mlp_dim is not correct.')

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)  # bs,tokens,channels
        y = self.ln(y.float())  # bs,tokens,channels
        y = torch.mean(y, dim=1, keepdim=False)  # bs,channels
        probs = self.fc(y)  # bs,num_classes
        return probs


if __name__ == '__main__':
    model = MlpMixer(num_classes=2,
                     num_blocks=6,
                     patch_size=6,
                     tokens_hidden_dim=128,
                     channels_hidden_dim=512,
                     tokens_mlp_dim=1,
                     max_len=501).cuda().to(torch.float32)
    from magnet_models2D import FPS
    from thop import profile
    a = torch.rand(1, 501).cuda().to(torch.float32)
    model(a)
    FPS(model, (a, ))
    flops, params = profile(model, inputs=(a, ), verbose=False)
    print()
    print('FLOPs = ' + str(flops / 1000**3) + 'G')
    print('Params = ' + str(params / 1000**2) + 'M')
    total_param = sum(p.numel() for p in model.parameters())
    train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model._get_name(), train_param, total_param)
