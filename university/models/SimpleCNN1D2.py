import torch
from torch import nn
from torch.nn import init


class SimpleCNN1D1(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_kernels, kernel_size, pool_size, n4, n_classes):
        super(SimpleCNN1D1, self).__init__()
        # [The first hidden convolution layer C1 filters the input_channels x 1 input data with 20 kernels of size k1 x 1]
        self.input_channels = input_channels
        self.conv = nn.Conv1d(1, n_kernels, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, n4)
        self.fc2 = nn.Linear(n4, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    net = SimpleCNN1D1(input_channels=204, n_kernels=20, kernel_size=10, pool_size=5, n4=100, n_classes=7)
    a = torch.rand(4, 204)
    print(net(a).shape)
    print(net(a))