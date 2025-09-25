# '''
# Description: 
# Date: 2023-07-21 14:36:27
# LastEditTime: 2023-07-27 18:41:47
# FilePath: /chengdongzhou/ScConv.py
# '''
# import torch
# import torch.nn.functional as F
# import torch.nn as nn 


# class GroupBatchnorm2d(nn.Module):
#     def __init__(self, c_num:int, 
#                  group_num:int = 16, 
#                  eps:float = 1e-10
#                  ):
#         super(GroupBatchnorm2d,self).__init__()
#         assert c_num    >= group_num
#         self.group_num  = group_num
#         self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
#         self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
#         self.eps        = eps
#     def forward(self, x):
#         N, C, H, W  = x.size()
#         x           = x.view(   N, self.group_num, -1   )
#         mean        = x.mean(   dim = 2, keepdim = True )
#         std         = x.std (   dim = 2, keepdim = True )
#         x           = (x - mean) / (std+self.eps)
#         x           = x.view(N, C, H, W)
#         return x * self.weight + self.bias


# class SRU(nn.Module):
#     def __init__(self,
#                  oup_channels:int, 
#                  group_num:int = 16,
#                  gate_treshold:float = 0.5,
#                  torch_gn:bool = True
#                  ):
#         super().__init__()
        
#         self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
#         self.gate_treshold  = gate_treshold
#         self.sigomid        = nn.Sigmoid()

#     def forward(self,x):
#         gn_x        = self.gn(x)
#         w_gamma     = self.gn.weight/sum(self.gn.weight)
#         w_gamma     = w_gamma.view(1,-1,1,1)
#         reweigts    = self.sigomid( gn_x * w_gamma )
#         # Gate
#         w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
#         w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
#         x_1         = w1 * x
#         x_2         = w2 * x
#         y           = self.reconstruct(x_1,x_2)
#         return y
    
#     def reconstruct(self,x_1,x_2):
#         x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
#         x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
#         return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''
#     def __init__(self, 
#                  op_channel:int,
#                  alpha:float = 1/2,
#                  squeeze_radio:int = 2 ,
#                  group_size:int = 2,
#                  group_kernel_size:int = 3,
#                  ):
#         super().__init__()
#         self.up_channel     = up_channel   =   int(alpha*op_channel)
#         self.low_channel    = low_channel  =   op_channel-up_channel
#         self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
#         self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
#         #up
#         self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
#         self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
#         #low
#         self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
#         self.advavg         = nn.AdaptiveAvgPool2d(1)

#     def forward(self,x):
#         # Split
#         up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
#         up,low  = self.squeeze1(up),self.squeeze2(low)
#         # Transform
#         Y1      = self.GWC(up) + self.PWC1(up)
#         Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
#         # Fuse
#         out     = torch.cat( [Y1,Y2], dim= 1 )
#         out     = F.softmax( self.advavg(out), dim=1 ) * out
#         out1,out2 = torch.split(out,out.size(1)//2,dim=1)
#         return out1+out2


# class ScConv(nn.Module):
#     def __init__(self,
#                 op_channel:int,
#                 group_num:int = 4,
#                 gate_treshold:float = 0.5,
#                 alpha:float = 1/2,
#                 squeeze_radio:int = 2 ,
#                 group_size:int = 2,
#                 group_kernel_size:int = 3,
#                  ):
#         super().__init__()
#         self.SRU = SRU( op_channel, 
#                        group_num            = group_num,  
#                        gate_treshold        = gate_treshold )
#         self.CRU = CRU( op_channel, 
#                        alpha                = alpha, 
#                        squeeze_radio        = squeeze_radio ,
#                        group_size           = group_size ,
#                        group_kernel_size    = group_kernel_size )
    
#     def forward(self,x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x


# if __name__ == '__main__':
#     x       = torch.randn(1,32,16,16)
#     model   = ScConv(32)
#     print(model(x).shape)
import torch
import torch.nn.functional as F
import torch.nn as nn 


class DropBlock2D(nn.Module):
    """
    简单的DropBlock实现，随机丢弃局部区域块
    """
    def __init__(self, drop_prob=0.1, block_size=3):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        N, C, H, W = x.size()
        
        # 计算gamma，使得实际丢弃的元素比例接近drop_prob
        gamma = self.drop_prob / (self.block_size ** 2)
        
        # 生成掩码（伯努利分布）
        mask = torch.bernoulli(torch.full((N, C, H - self.block_size + 1, W - self.block_size + 1), gamma, device=x.device))
        
        # 扩展掩码为block_size x block_size的块
        mask = F.pad(mask, [self.block_size//2] * 4, mode='constant', value=0)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask  # 反转掩码，1表示保留，0表示丢弃
        
        # 应用掩码并归一化
        x = x * mask
        x = x / mask.mean()  # 保持期望不变
        
        return x


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True,
                 # 添加DropBlock参数
                 dropblock_prob: float = 0.1,
                 dropblock_size: int = 3
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()
        
        # 初始化DropBlock层，放在特征重组之后
        self.dropblock = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size)

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        
        # Gate操作
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)
        x_1         = w1 * x
        x_2         = w2 * x
        
        # 特征重组
        y           = self.reconstruct(x_1,x_2)
        
        # 在重组后添加DropBlock
        y = self.dropblock(y)
        
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                # 传递DropBlock参数
                dropblock_prob: float = 0.1,
                dropblock_size: int = 3
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold,
                       dropblock_prob       = dropblock_prob,  # 传递给SRU
                       dropblock_size       = dropblock_size   # 传递给SRU
                       )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


if __name__ == '__main__':
    x       = torch.randn(1,32,16,16)
    model   = ScConv(32, dropblock_prob=0.5, dropblock_size=3)  # 可调整DropBlock参数
    print(model(x).shape)  # 输出: torch.Size([1, 32, 16, 16])
