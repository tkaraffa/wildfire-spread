import torch
import torch.nn as nn
import torch.nn.functional as F


def reweight(cls_num_list, beta=0.9999):
    per_cls_weights = torch.Tensor(
        list(map(lambda n: (1 - beta) / (1 - beta**n), cls_num_list))
    )
    per_cls_weights *= len(cls_num_list) / per_cls_weights.sum()
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return F.cross_entropy(
            (1 - F.softmax(input, dim=1)) ** self.gamma * F.log_softmax(input, dim=1),
            target,
            weight=self.weight,
        )
