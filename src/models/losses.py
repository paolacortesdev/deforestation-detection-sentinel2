import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (2. * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def get_combined_loss(pos_weight=2.0):

    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight])
    )

    dice = DiceLoss()

    def loss_fn(pred, target):
        return bce(pred, target) + dice(pred, target)

    return loss_fn