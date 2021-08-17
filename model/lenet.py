import torch 
import torch.nn as nn
import torch.nn.functional as F 

def conv_layer(ic, oc, ks=3, stride=1, padding=1, use_bn=True, use_activ=True, drop_rate=0.5):
    layer = nn.Sequential()
    layer.add_module('conv', nn.Conv2d(ic, oc, ks, stride, padding))
    if use_bn: layer.add_module('bn', nn.BatchNorm2d(oc))
    if use_activ: layer.add_module('relu', nn.ReLU(inplace=True))
    if drop_rate > 0: layer.add_module('drop', nn.Dropout2d(drop_rate))
    return layer

# SVHN(32) -> MNIST(28->32)
class LeNet3(nn.Module):
    def __init__(self, num_classes, inplanes=3, **kwargs):
        super().__init__()
        self.layer1 = conv_layer(inplanes, 64, ks=5, stride=2, padding=2, drop_rate=0.1)
        self.layer2 = conv_layer(64, 128, ks=5, stride=2, padding=2, drop_rate=0.3)
        self.layer3 = conv_layer(128, 256, ks=5, stride=2, padding=2, drop_rate=0.5)

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Linear(512, num_classes)
        
        self.inplanes = 512

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        self.features = x
        x = self.fc2(x)
        return x

# MNIST(28) <-> USPS(16->28)
class LeNet2(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500), 
            nn.ReLU(), 
            nn.Dropout(p=0.5)
        )
        
        self.fc2 = nn.Linear(500, num_classes)
        self.inplanes = 500


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        self.features = x
        x = self.fc2(x)
        return x

class LeNet21(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(100, num_classes)
        self.inplanes = 100

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), 48*4*4)
        x = F.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        self.features = x
        x = self.fc3(x)
        return x
