from torch.autograd import Function
from functools import reduce

import torch
import torch.nn as nn


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class RandomizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        x = torch.mm(x, self.weight)
        return x


class RandomizedMultiLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.ModuleList(
            [RandomizedLinear(nfts, out_features) for nfts in in_features])
        self.norm = out_features ** (1 / len(in_features))

    def forward(self, x):
        x = [m(xx) for m, xx in zip(self.weight, x)]
        x = reduce(torch.mul, x)
        x = x / self.norm
        return x


def batch_multi_linear(x):
    feature, score = x
    bs = score.size(0)
    score = score.unsqueeze(1)
    feature = feature.unsqueeze(2)
    x = torch.bmm(feature, score)
    return x.view(bs, -1).contiguous()


class MultiLinear(nn.Module):
    def __init__(self, randomized=False, in_features=None, out_features=None):
        super().__init__()
        self.randomized = randomized
        if self.randomized:
            self.bilinear = RandomizedMultiLinear(in_features, out_features)
        else:
            self.bilinear = batch_multi_linear

    def forward(self, x):
        x = self.bilinear(x)
        return x


def gradient_reverse(x, coef=1.0):
    def func(coef):
        return lambda grad: -coef * grad.clone()
    x.register_hook(func(coef))


class GradientReverseLayer(nn.Module):
    def forward(self, x, coef=1):
        x=x*1
        gradient_reverse(x, coef)
        return x
