import numpy as np
import torch as th

class DiceLoss(th.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = th.sum(score * target)
        y_sum = th.sum(target * target)
        z_sum = th.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        if weight is None:
            # weight = [1] * self.n_classes
            weight = [0.1, 1.1, 1.0, 1.0, 1.5, 1.0, 1.1, 1.0, 1.5]
        weight = th.tensor(weight).to(inputs.device)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / weight.sum()
        # return loss / self.n_classes
    
class FocalLoss(th.nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = th.tensor([0.1, 1.1, 1.0, 1.0, 1.5, 1.0, 1.1, 1.0, 1.5])
        # self.weight = weight

    def forward(self, inputs, targets):
        self.weight = self.weight.to(inputs.device)
        ce_loss = th.nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = th.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


def preprocess_output_for_diceloss(output):
    output = th.softmax(output, dim=1)
    return output

def preprocess_output_for_celoss(output):

    return output

def preprocess_target_for_loss(target):
    target = th.argmax(target, dim=1)
    return target   

