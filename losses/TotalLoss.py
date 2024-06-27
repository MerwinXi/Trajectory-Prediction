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
        # print('pred:',pred.shape)
        # print('truth:',truth.shape)
        abs_diff = torch.abs(pred - truth)
        quadratic = torch.where(abs_diff <= delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))
        return quadratic.mean()

    def daa_regression_loss(self, pred, truth, drivable_mask, delta=1.0):
        """可行驶区域感知的回归损失，增加额外的惩罚项。"""
        base_loss = self.huber_loss(pred, truth, delta)
        # 计算预测点与真实点的欧式距离
        DE = torch.sqrt(torch.sum((pred - truth) ** 2, dim=1))
        DE = torch.reshape(DE,(DE.shape[0],-1,2))
        # print("DE shape:", DE.shape)  # 输出 DE 的尺寸以进行检查
        # print("drivable_mask:", drivable_mask.shape)  # 输出 DE 的尺寸以进行检查
        if drivable_mask.dim() == 2:
            drivable_mask = drivable_mask.unsqueeze(1)
        # print("drivable_mask shape after unsqueeze:", drivable_mask.shape)  # 输出 drivable_mask 的尺寸以进行检查
        # print('DE:',DE.shape)
        # 确保 drivable_mask 的最后一个维度与 DE 的第一个维度匹配
        # if drivable_mask.size(-2) != DE.size(-2):
        #     # 调整 drivable_mask 的尺寸
        #     DE = DE.expand(DE.size(0), drivable_mask.size(1), DE.size(2))
        #     print("DE shape:", DE.shape)  # 输出调整后的 drivable_mask 的尺寸
        DE = DE.unsqueeze(1).expand(-1, drivable_mask.shape[1], -1, -1)

        # 检查 a_expanded 的形状
        # print("Shape of a_expanded:", DE.shape)

        # 确保 a_expanded 和 b 可以相乘
        # 使用 unsqueeze 增加一个维度，以便进行广播
        drivable_mask_expanded = drivable_mask.unsqueeze(2)
        penalty = torch.log(1 + DE) * drivable_mask_expanded  # 执行计算

        return (base_loss + penalty).mean()

    def confidence_loss(self, pred, truth):
        """使用KL散度计算预测和真实分布之间的置信度损失。"""
        pred_log_softmax = F.log_softmax(pred, dim=1)
        truth_softmax = F.softmax(truth, dim=1)
        return F.kl_div(pred_log_softmax, truth_softmax, reduction='batchmean')

    def classification_loss(self, pred, labels):
        """分类任务的交叉熵损失。"""
        # print('pred:',pred.shape,'labels:',labels.shape)
        return F.cross_entropy(pred[:,:,-2:], labels[:,:,-2:])

    def forward(self, pred, truth, drivable_mask, labels):
        """计算总损失，结合三个损失及其权重。"""
        # print(pred[0], truth[0])
        l_daa_reg = self.daa_regression_loss(pred, truth, drivable_mask)
        l_conf = self.confidence_loss(pred, truth)
        l_cls = self.classification_loss(pred, labels)

        # 每个权重参数的对数正则项
        regularization = torch.log(self.alpha1 + 1) + torch.log(self.alpha2 + 1) + torch.log(self.alpha3 + 1)

        # 加权求和的总损失
        # l_sum = (1 / self.alpha1 ** 2) * l_daa_reg + \
        #         (1 / self.alpha2 ** 2) * l_conf + \
        #         (1 / self.alpha3 ** 2) * l_cls + \
        #         regularization
        # print(l_daa_reg.item(),l_conf.item(),l_cls.item())
        # return l_sum
        return l_daa_reg + l_conf 
