import torch.nn as nn
import torchvision
import model.backbone as backbone
import torch

def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.ones((1, ns)) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

# class MMD_loss(nn.Module):
#     def __init__(self, kernel_type='liner', kernel_mul=2.0, kernel_num=5):
#         super(MMD_loss, self).__init__()
#         self.kernel_num = kernel_num
#         self.kernel_mul = kernel_mul
#         self.fix_sigma = None
#         self.kernel_type = kernel_type

#     def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#         total0 = total.unsqueeze(0).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         total1 = total.unsqueeze(1).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         L2_distance = ((total0-total1)**2).sum(2)
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul**i)
#                           for i in range(kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
#                       for bandwidth_temp in bandwidth_list]
#         return sum(kernel_val)

#     def linear_mmd2(self, f_of_X, f_of_Y):
#         loss = 0.0
#         delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
#         loss = delta.dot(delta.T)
#         return loss

#     def mmd_linear(self, f_of_X, f_of_Y):
#         delta = f_of_X - f_of_Y
#         loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
#         return loss

#     def forward(self, source, target):
#         if self.kernel_type == 'linear':
#             return self.mmd_linear(source, target)
#         elif self.kernel_type == 'rbf':
#             batch_size = int(source.size()[0])
#             kernels = self.guassian_kernel(
#                 source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             with torch.no_grad():
#                 XX = torch.mean(kernels[:batch_size, :batch_size])
#                 YY = torch.mean(kernels[batch_size:, batch_size:])
#                 XY = torch.mean(kernels[:batch_size, batch_size:])
#                 YX = torch.mean(kernels[batch_size:, :batch_size])
#                 loss = torch.mean(XX + YY - XY - YX)
#             torch.cuda.empty_cache()
#             return loss


class DDCNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(DDCNet, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width),nn.ReLU()]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(1):
            self.classifier_layer[i].weight.data.normal_(0, 0.01)
            self.classifier_layer[i].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target)
        source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
        return source_clf, transfer_loss

    def predict(self, x):
        features = self.base_network(x)
        self.features = features
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            delta = X - Y
            loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
            # mmd_loss = MMD_loss()
            # loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss
