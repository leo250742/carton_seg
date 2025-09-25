import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Concat_bifpn(nn.Module):
    # 带尺寸调整的特征融合模块
    def __init__(self, c1, c2):
        super(Concat_bifpn, self).__init__()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4
        self.conv = Conv(c1, c2, 1, 1, 0)
        self.act = nn.ReLU()
 
    def forward(self, x):
        if not isinstance(x, list):
            return x
            
        # 确保所有输入张量具有相同的空间尺寸
        # 以第一个张量的尺寸为基准进行调整
        target_size = x[0].shape[2:]  # 获取H和W维度
        adjusted_x = []
        
        for tensor in x:
            # 如果尺寸不匹配，则进行上采样或下采样
            if tensor.shape[2:] != target_size:
                # 使用双线性插值调整尺寸
                adjusted = F.interpolate(
                    tensor, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                adjusted_x.append(adjusted)
            else:
                adjusted_x.append(tensor)
        
        # 进行加权融合
        if len(adjusted_x) == 2:
            w = self.w1
            weight = w / (torch.sum(w) + self.epsilon)
            x = self.conv(self.act(weight[0] * adjusted_x[0] + weight[1] * adjusted_x[1]))
        elif len(adjusted_x) == 3: 
            w = self.w2
            weight = w / (torch.sum(w) + self.epsilon)
            x = self.conv(self.act(weight[0] * adjusted_x[0] + weight[1] * adjusted_x[1] + weight[2] * adjusted_x[2]))
        elif len(adjusted_x) == 4:    
            w = self.w3
            weight = w / (torch.sum(w) + self.epsilon)
            x = self.conv(self.act(weight[0] * adjusted_x[0] + weight[1] * adjusted_x[1] + 
                                  weight[2] * adjusted_x[2] + weight[3] * adjusted_x[3]))
        else:
            raise ValueError(f"Concat_bifpn expects 2, 3, or 4 inputs, got {len(adjusted_x)}")
            
        return x