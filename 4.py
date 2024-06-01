import torch
import torch.nn as nn

class DAALoss(nn.Module):
    def __init__(self):
        super(DAALoss, self).__init__()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, pred, target, in_drivable_area, distance_error):
        huber_loss = self.huber_loss(pred, target)
        penalty = torch.log1p(distance_error)
        loss = torch.where(in_drivable_area, huber_loss, huber_loss * penalty)
        return loss.mean()

class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred_dist, target_dist):
        return self.kl_div(pred_dist.log(), target_dist)

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.cross_entropy(pred, target)

class TotalLoss(nn.Module):
    def __init__(self, alpha1, alpha2, alpha3, alpha_aux):
        super(TotalLoss, self).__init__()
        self.daa_loss = DAALoss()
        self.conf_loss = ConfidenceLoss()
        self.cls_loss = ClassificationLoss()
        self.alpha1 = nn.Parameter(torch.tensor(alpha1, requires_grad=True))
        self.alpha2 = nn.Parameter(torch.tensor(alpha2, requires_grad=True))
        self.alpha3 = nn.Parameter(torch.tensor(alpha3, requires_grad=True))
        self.alpha_aux = nn.Parameter(torch.tensor(alpha_aux, requires_grad=True))  # 辅助损失权重

    def forward(self, pred, target, in_drivable_area, distance_error, pred_dist, target_dist, cls_pred, cls_target, auxiliary_output, auxiliary_target):
        loss_daa = self.daa_loss(pred, target, in_drivable_area, distance_error)
        loss_conf = self.conf_loss(pred_dist, target_dist)
        loss_cls = self.cls_loss(cls_pred, cls_target)
        loss_aux = self.cls_loss(auxiliary_output, auxiliary_target)  # 辅助损失

        total_loss = (1 / (2 * self.alpha1**2)) * loss_daa + \
                     (1 / (2 * self.alpha2**2)) * loss_conf + \
                     (1 / (2 * self.alpha3**2)) * loss_cls + \
                     (1 / (2 * self.alpha_aux**2)) * loss_aux + \
                     torch.log(1 + self.alpha1) + \
                     torch.log(1 + self.alpha2) + \
                     torch.log(1 + self.alpha3) + \
                     torch.log(1 + self.alpha_aux)

        return total_loss.mean()


