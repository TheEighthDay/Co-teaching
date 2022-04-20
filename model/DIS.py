import torch.nn as nn
import torchvision
import model.backbone as backbone
import torch

class StudentNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', width=1024):
        super(StudentNet, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        classifier_layer_list = [nn.Linear(2048, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)
        # for i in range(2):
        #     self.classifier_layer[i].weight.data.normal_(0, 0.01)
        #     self.classifier_layer[i].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.base_network(x)
        self.features=x
        x= self.classifier_layer(x)
        return x

