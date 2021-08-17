import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import model.backbone as backbone

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Extractor(nn.Module):

    def __init__(self,num_classes=65, base_net='ResNet50'):
        super(Extractor, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()

    def forward(self, x):
        x=self.sharedNet(x)
        return x

class Class_classifier(nn.Module):

    def __init__(self,num_classes, base_net='ResNet50'):
        super(Class_classifier, self).__init__()
        self.source_fc = nn.Linear(2048, num_classes)

        self.source_fc.weight.data.normal_(0, 0.01)
        self.source_fc.bias.data.fill_(0.0)


    def forward(self, x):
        x=self.source_fc(x)
        return x

class Domain_classifier(nn.Module):

    def __init__(self,num_classes, base_net='ResNet50'):
        super(Domain_classifier, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(2048, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = self.domain_classifier(x)

        return x

class DANNNet(nn.Module):
    def __init__(self,num_classes, base_net='ResNet50'):
        super(DANNNet, self).__init__()
        self.ExtractorNet = Extractor(num_classes=num_classes,base_net=base_net)
        self.Class_classifierNet = Class_classifier(num_classes=num_classes,base_net=base_net)
        self.Domain_classifierNet = Domain_classifier(num_classes=num_classes,base_net=base_net)

    def forward(self, source_data,target_data,constant):
        src_feature = self.ExtractorNet(source_data)
        tgt_feature = self.ExtractorNet(target_data)

        class_preds = self.Class_classifierNet(src_feature)

        tgt_preds = self.Domain_classifierNet(tgt_feature, constant)
        src_preds = self.Domain_classifierNet(src_feature, constant)
        return class_preds,tgt_preds,src_preds





