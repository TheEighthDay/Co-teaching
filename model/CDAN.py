import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import MultiLinear, gradient_reverse
from model.resnet import resnet34, resnet50
from model.discriminator import NLayerDiscriminator, MultiNLayerDiscriminator
from model.lenet import LeNet2, LeNet21, LeNet3

net = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'LeNet2': LeNet2,
    'LeNet21': LeNet21,
    'LeNet3': LeNet3
}

class CDAN(nn.Module):
    def __init__(
        self,
        num_classes,
        randomized=False,
        D_hidden_size=1024,
        multi_linear_out_features=2048,
        fc_hidden_dim=None,
        backbone='resnet50',
        pretrained=True
    ):
        super().__init__()
        self.net_G = net[backbone](num_classes=num_classes, pretrained=pretrained, fc_hidden_dim=fc_hidden_dim)
        self.multi_linear = MultiLinear(
            randomized=randomized,
            in_features=[self.net_G.inplanes, num_classes],
            out_features=multi_linear_out_features
        )
        self.net_D = NLayerDiscriminator(
            in_feature=multi_linear_out_features if randomized else self.net_G.inplanes * num_classes , 
            hidden_size=D_hidden_size
        )


    def forward(self, x, coef=1.0):
        outputs = self.net_G(x)
        features = self.net_G.features

        scores = torch.softmax(outputs, dim=1)
        multi_linear_feat = self.multi_linear([features, scores.detach()])

        out_D = self.net_D(multi_linear_feat, coef)

        return outputs, out_D


class CDAN_Multi(nn.Module):
    def __init__(
        self,
        num_classes,
        num_domains,
        randomized=False,
        D_hidden_size=1024,
        multi_linear_out_features=2048,
        fc_hidden_dim=None,
        backbone='resnet50',
        pretrained=True
    ):
        super().__init__()
        self.net_G = net[backbone](num_classes=num_classes, pretrained=pretrained, fc_hidden_dim=fc_hidden_dim)
        self.multi_linear = MultiLinear(
            randomized=randomized,
            in_features=[self.net_G.inplanes, num_classes],
            out_features=multi_linear_out_features
        )
        self.net_D = MultiNLayerDiscriminator(
            in_feature=multi_linear_out_features if randomized else self.net_G.inplanes * num_classes , 
            hidden_size=D_hidden_size,
            num_classes=num_domains
        )


    def forward(self, x, coef=1.0):
        outputs = self.net_G(x)
        features = self.net_G.features

        scores = torch.softmax(outputs, dim=1)
        multi_linear_feat = self.multi_linear([features, scores.detach()])

        out_D = self.net_D(multi_linear_feat, coef)

        return outputs, out_D

class CDAN_2D(nn.Module):
    def __init__(
        self,
        num_classes,
        randomized=False,
        D_hidden_size=1024,
        multi_linear_out_features=2048,
        fc_hidden_dim=None,
        backbone='resnet50'
    ):
        super().__init__()
        self.net_G = net[backbone](num_classes=num_classes, pretrained=True, fc_hidden_dim=fc_hidden_dim)
        self.multi_linear = MultiLinear(
            randomized=randomized,
            in_features=[self.net_G.inplanes, num_classes],
            out_features=multi_linear_out_features
        )
        self.net_D = NLayerDiscriminator(
            in_feature=multi_linear_out_features if randomized else self.net_G.inplanes * num_classes , 
            hidden_size=D_hidden_size
        )


    def forward(self, x, coef=1.0):
        outputs = self.net_G(x)
        features = self.net_G.features

        scores = torch.softmax(outputs, dim=1)
        multi_linear_feat = self.multi_linear([features, scores.detach()])

        out_D = self.net_D(multi_linear_feat, coef)

        return outputs, out_D, features
