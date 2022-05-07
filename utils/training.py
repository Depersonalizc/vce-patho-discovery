from torchvision import transforms
from torch import nn
import torch
import torch.nn.functional as F


RESNET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def want_grad(module, want):
    for param in module.parameters():
        param.requires_grad = want


def poly_lr_scheduler(optimizer, base_lr, max_iter, power=0.9, **kwargs):
    lr_lambda = lambda it: base_lr * ((1 - it / max_iter) ** power)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda], **kwargs)


def bin_prob_to_cls_entropy(prob):
    # prob: (B, 1, H, W) binary probability map
    # return: (B, 2, H, W) entropy map factored into 2 classes
    two_cls = torch.cat([prob, 1 - prob], 1)  # (B, 2, H, W)
    return SelfInformation()(two_cls)


def one_cls_bce(y_pred, y_label):
    y_gt = torch.zeros(y_pred.size(), dtype=torch.float32, device=y_pred.device)
    y_gt.fill_(y_label)
    return F.binary_cross_entropy_with_logits(y_pred, y_gt)


def log_tensorboard(writer, entry_dict, i_iter, section='train'):
    for name, value in entry_dict.items():
        writer.add_scalar(f'{section}/{name}', value, i_iter)


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

