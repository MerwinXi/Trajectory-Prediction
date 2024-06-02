import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLossModule(nn.Module):
    def __init__(self):
        super(CustomLossModule, self).__init__()
        # 可学习的权重参数
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.alpha3 = nn.Parameter(torch.tensor(1.0))

    def huber_loss(self, pred, truth, delta=1.0):
        """标准的Huber损失，用于平滑回归损失计算。"""
        abs_diff = torch.abs(pred - truth)
        quadratic = torch.where(abs_diff <= delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))
        return quadratic.mean()

    def daa_regression_loss(self, pred, truth, drivable_mask, delta=1.0):
        """可行驶区域感知的回归损失，增加额外的惩罚项。"""
        base_loss = self.huber_loss(pred, truth, delta)
        # 计算预测点与真实点的欧式距离
        DE = torch.sqrt(torch.sum((pred - truth) ** 2, dim=1))
        # 应用条件惩罚
        penalty = torch.log(1 + DE) * drivable_mask
        return (base_loss + penalty).mean()

    def confidence_loss(self, pred, truth):
        """使用KL散度计算预测和真实分布之间的置信度损失。"""
        pred_log_softmax = F.log_softmax(pred, dim=1)
        truth_softmax = F.softmax(truth, dim=1)
        return F.kl_div(pred_log_softmax, truth_softmax, reduction='batchmean')

    def classification_loss(self, pred, labels):
        """分类任务的交叉熵损失。"""
        return F.cross_entropy(pred, labels)

    def forward(self, pred, truth, drivable_mask, labels):
        """计算总损失，结合三个损失及其权重。"""
        l_daa_reg = self.daa_regression_loss(pred, truth, drivable_mask)
        l_conf = self.confidence_loss(pred, truth)
        l_cls = self.classification_loss(pred, labels)

        # 每个权重参数的对数正则项
        regularization = torch.log(self.alpha1 + 1) + torch.log(self.alpha2 + 1) + torch.log(self.alpha3 + 1)

        # 加权求和的总损失
        l_sum = (1 / self.alpha1 ** 2) * l_daa_reg + \
                (1 / self.alpha2 ** 2) * l_conf + \
                (1 / self.alpha3 ** 2) * l_cls + \
                regularization

        return l_sum

