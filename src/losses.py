from torch import nn
import torch


class ContrastiveLoss_yl(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss_yl, self).__init__()
        self.margin = margin
        self.i = 0

    def forward(self, inspection, reference, label):
        dist = torch.sqrt(torch.pow((inspection-reference),2).sum(dim=1).sum(dim=1).sum(dim=1))
        
        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))

        return loss