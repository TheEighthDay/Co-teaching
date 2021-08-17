import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def cross_entropy_loss(input, target=None, pseudo=False, temperature=None):
    # target is unknown, maximize the largest logit
    if target is None:
        target = torch.argmax(input, dim=1)
        loss = F.cross_entropy(input, target)

    # softmax
    if target is not None and target.dtype == torch.int64:
        loss = F.cross_entropy(input, target)    

    # soft label is known, use the class of the highest score as the target
    if target is not None and target.dtype != torch.int64 and pseudo is True:
        target = target.argmax(dim=1)
        loss = F.cross_entropy(input, target)
    
    # soft label softmax
    if target is not None and target.dtype != torch.int64 and pseudo is False:
        if temperature is not None:
            input = input / temperature
            target = target / temperature
        log_input = torch.log_softmax(input, dim=1)
        loss = -(log_input * target).sum(1).mean()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def mixup_data(sourcex,targetx, sourcey,targety):
    lam = np.random.beta(1, 1)
    mixed_x = lam * sourcex + (1 - lam) * targetx
    mixed_y = lam * sourcey + (1 - lam) * targety
    return mixed_x, mixed_y

if __name__ == '__main__':
    from model import DDC
    print("ss")


