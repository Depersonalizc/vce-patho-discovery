from torchvision import transforms
from torch import nn
import torch

RESNET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def stop_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def poly_lr_scheduler(optimizer, base_lr, max_iter, power=0.9, **kwargs):
    lr_lambda = lambda it: base_lr * ((1 - it / max_iter) ** power)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda], **kwargs)


class SelfInformation(nn.Module):
    def __init__(self, eps=1e-30):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x = x + self.eps
        return -x * torch.log(x)


class BinaryEntropy(nn.Module):
    def __init__(self, eps=1e-30, mean_reduce=True):
        super().__init__()
        self.info = SelfInformation(eps)
        self.mean_reduce = mean_reduce

    def forward(self, p):
        # p: probability map whose elements are from 0 to 1
        entropy = self.info(p) + self.info(1 - p)
        if self.mean_reduce:
            return entropy.mean()
        return entropy

