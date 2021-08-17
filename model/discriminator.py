import numpy as np
import torch
import torch.nn as nn

from model.utils import gradient_reverse


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, coef=1.0):
        if self.training:
            gradient_reverse(x, coef)

        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1


class MultiNLayerDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.ad_layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, coef=1.0):
        if self.training:
            gradient_reverse(x, coef)

        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return self.num_classes