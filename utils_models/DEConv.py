import torch
import torch.nn as nn


class DEConv_2(nn.Module):
    def __init__(self, dim):
        super(DEConv_2, self).__init__()

        self.conv1_1 = nn.Conv2d(dim, dim, 3, bias=True)
        self.conv1_2 = nn.Conv2d(dim, dim, 3, bias=True)
        self.conv1_3 = nn.Conv2d(dim, dim, 3, bias=True)
        self.conv1_4 = nn.Conv2d(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(dim * 5, dim, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        bach_shape = x.shape[0]
        x = x.view(bach_shape, 32, 32, 1)
        w1, b1 = self.conv1_1.weight, self.conv1_1.bias
        w2, b2 = self.conv1_2.weight, self.conv1_2.bias
        w3, b3 = self.conv1_3.weight, self.conv1_3.bias
        w4, b4 = self.conv1_4.weight, self.conv1_4.bias
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w1 = w1 + w2 + w3 + w4 + w5
        b = b1 + b2 + b3 + b4 + b5

        w2 = torch.cat([w1, w2, w3, w4, w5], dim=1)
        w2 = self.conv1(w2)
        res1 = nn.functional.conv2d(input=x, weight=w1, bias=b, stride=1, padding=1, groups=1)
        res2 = nn.functional.conv2d(input=x, weight=w2, bias=b, stride=1, padding=1, groups=1)

        a = self.sigmod(res1 + res2)

        out = x + a * res1 + (1 - a) * res2
        out = out.view(bach_shape, 1024)
        return out