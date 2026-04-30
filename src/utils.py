import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mixup_data(x, y, alpha=0.2):
    """返回混合后的图像和标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def label_smoothing_loss(pred, target, smoothing=0.1):
    """标签平滑交叉熵"""
    n_class = pred.size(1)
    log_pred = F.log_softmax(pred, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (n_class - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return torch.mean(torch.sum(-true_dist * log_pred, dim=1))