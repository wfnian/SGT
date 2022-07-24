import torch
from torch import nn


class SimpleCNN1D(nn.Module):
    def __init__(self, n_feature, n_output, n_cnn_kernel, n_mult_factor=2, pool=False):
        super(SimpleCNN1D, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_feature * n_mult_factor
        self.n_output = n_output
        self.n_cnn_kernel = n_cnn_kernel
        self.n_mult_factor = n_mult_factor
        self.n_l2_hidden = self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3)
        #         self.n_out_hidden=int (self.n_l2_hidden/2)

        self.l1 = nn.Sequential(torch.nn.Linear(self.n_feature, self.n_hidden), torch.nn.Dropout(p=1 - .85),
                                torch.nn.LeakyReLU(0.1),
                                torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True))
        self.c1 = nn.Sequential(
            torch.nn.Conv1d(self.n_feature,
                            self.n_hidden,
                            kernel_size=(self.n_cnn_kernel, ),
                            stride=(1, ),
                            padding=(1, )), torch.nn.Dropout(p=1 - .75), torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True),
            torch.nn.MaxPool1d(kernel_size=(3, ), stride=1, padding=1) if pool else torch.nn.Identity())
        self.out = nn.Sequential(torch.nn.Linear(self.n_l2_hidden, self.n_output), )
        self.sig = nn.Softmax()

    def forward(self, x):
        varSize = x.data.shape[0]  # must be calculated here in forward() since its is a dynamic size
        x = self.l1(x)
        x = x.view(varSize, self.n_feature, self.n_mult_factor)
        x = self.c1(x)
        # for Linear layer
        x = x.view(varSize, self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3))
        x = self.out(x)
        x = self.sig(x)
        return x


if __name__ == '__main__':

    net = SimpleCNN1D(n_feature=31, n_output=3, n_cnn_kernel=3)
    a = torch.rand(4, 31)
    print(net(a).shape)
    print(net(a))