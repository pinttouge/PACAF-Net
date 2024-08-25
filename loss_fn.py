import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidentLoss:
    def __init__(self, lmbd=3):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.weight = [0.7, 0.7, 0.7, 0.7,
                       0.5, 0.7, 0.9, 1.3]

        self.lmbda = float(int(lmbd) / 10)

    def weighted_bce(self, pred, gt):
        weit = 1 + 4 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        wbce = (self.bce(pred, gt) * weit).sum(dim=[2, 3]) / weit.sum(dim=[2, 3])
        return wbce.mean()

    def confident_loss(self, pred, gt, beta=2):
        y = torch.sigmoid(pred)
        weight = beta * y * (1 - y)
        weight = weight.detach()
        loss = (self.bce(pred, gt) * weight).mean()
        loss2 = self.lmbda * beta * (y * (1 - y)).mean()
        return loss + loss2

    def ce_loss(self, pred, gt):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()

    def get_value(self, X, sal_gt):
        sal_loss = 0
        count = 0

        for sal_pred, wght in zip(X, self.weight):
            scale = int(sal_gt.size(-1) / sal_pred.size(-1))
            target = sal_gt.gt(0.5).float()
            if count <= 3:  # global context
                target = F.pixel_unshuffle(target, scale)
                stage_sal_loss = self.weighted_bce(sal_pred, target)
            else:
                if scale > 1:
                    sal_pred = F.pixel_shuffle(sal_pred, scale)
                    # sal_pred = F.interpolate(sal_pred, size=(256,256), mode='bilinear', align_corners=False)
                stage_sal_loss = self.weighted_bce(sal_pred, target)
                if count % 3 == 0:
                    stage_sal_loss += self.confident_loss(sal_pred, target, beta=2)
            sal_loss += wght * stage_sal_loss

            count += 1

        return sal_loss
