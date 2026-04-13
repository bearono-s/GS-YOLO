import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from .conv import Conv, autopad


class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 极简双路径卷积
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, 1, 1, 0, groups=min(dim, 8))
        
        # 简化归一化
        self.norm = nn.GroupNorm(2, dim)  # 固定2组
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()
        
        # 极简注意力机制：直接融合avg和max
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 单层卷积融合
        self.fusion_conv = nn.Conv2d(2, dim, 1, bias=False)  # 直接映射到通道数

    def _calculate_local_saliency(self, x):
        """局部显著性统计量"""
        B, C, H, W = x.shape
        
        # 使用平均池化计算局部对比度
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        local_saliency = torch.abs(x - local_mean).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        return self.avg_pool(local_saliency)  # [B, 1, 1, 1]

    def forward(self, x):
        local_feat = self.dwconv(x)
        cross_feat = self.pwconv(x)
        
        alpha = self.sigmoid(self.alpha)
        diff_feat = local_feat - alpha * cross_feat
        diff_feat = self.norm(diff_feat)
        
        # 计算avg和max
        avg_val = self._calculate_local_saliency(diff_feat)  # [B, C, 1, 1]
        max_val = self.max_pool(diff_feat)  # [B, C, 1, 1]
        
        # 转换为单通道特征
        avg_single = torch.mean(avg_val, dim=1, keepdim=True)  # [B, 1, 1, 1]
        max_single = torch.mean(max_val, dim=1, keepdim=True)  # [B, 1, 1, 1]
        
        # 拼接并生成权重
        combined = torch.cat([avg_single, max_single], dim=1)  # [B, 2, 1, 1]
        weight = self.fusion_conv(combined)  # [B, C, 1, 1]
        weight = self.sigmoid(weight)
        
        return weight


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gaussion = Gaussian(dim, 7, 0.5)
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gaussion(x)
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6


class GDCM(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class GDCM_button(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=autopad(self.kernel_conv), stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        # scaling = (self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride


        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out
        
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class LOGStem(nn.Module):

    def __init__(self, in_chans, stem_dim):
        super().__init__()
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.down1 = nn.Sequential(
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c12),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(out_c12, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
        )
        # 定义LoG滤波器
        self.LoG = LoGFilter(in_chans, out_c12, 7, 1.0)
        # gaussian
        self.gaussian = Gaussian(out_c12, 7, 0.5)

    def forward(self, x):
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.down1(x)
        x = x + self.gaussian(x)
        x = self.down2(x)
        return x  # x = [B, C, H/4, W/4]

class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        # 保存sigma参数，确保在FP32精度下
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        self.kernel_size = kernel_size
        """创建高斯-拉普拉斯核"""
        # 初始化二维坐标 - 在FP32精度下计算
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        # 计算高斯-拉普拉斯核 - 在FP32精度下计算
        sigma_sq = self.sigma ** 2
        xx_yy_sq = xx**2 + yy**2
        kernel = (xx_yy_sq - 2 * sigma_sq) / (2 * math.pi * sigma_sq ** 2) * torch.exp(-xx_yy_sq / (2 * sigma_sq))
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum()  # 使用绝对值求和确保稳定
        log_kernel = kernel.unsqueeze(0).unsqueeze(0) # 添加 batch 和 channel 维度
        # 使用depthwise卷积，确保权重在FP32精度下初始化
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, 
                            padding=kernel_size // 2, groups=out_c, bias=False)
        # 设置权重并确保为FP32精度
        with torch.no_grad():
            self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1).to(torch.float32)
        # 锁定LoG权重，不参与训练（可选，根据需求）
        # self.LoG.weight.requires_grad = False
        self.act = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        # 初始卷积可以在混合精度下进行
        x = self.conv_init(x)  # x = [B, C/4, H, W]
        # LoG卷积需要在FP32精度下进行以确保数值稳定性
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入转换为FP32进行LoG计算
            x_fp32 = x.to(torch.float32)
            LoG = self.LoG(x_fp32)
            # 批归一化和激活函数可以在混合精度下进行
            LoG_edge = self.act(self.norm1(LoG))
        x = x + LoG_edge
        return x
    def half(self):
        """重写half方法以确保LoG权重保持FP32精度"""
        super().half()
        # 确保LoG权重保持FP32
        self.LoG.weight.data = self.LoG.weight.data.to(torch.float32)
        return self

class Gaussian(nn.Module):
    def __init__(self, dim, size, sigma):
        super().__init__()
        # 保存参数
        self.dim = dim
        self.size = size
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
        # 预先计算高斯核
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(size, sigma))
        # 创建卷积层 - 设置为depthwise卷积
        self.gaussian_conv = nn.Conv2d(dim, dim, kernel_size=size, stride=1, 
                                     padding=size // 2, groups=dim, bias=False)
        # 初始化权重并锁定
        with torch.no_grad():
            self.gaussian_conv.weight.data = self.gaussian_kernel.repeat(dim, 1, 1, 1)
        self.gaussian_conv.weight.requires_grad = False
        # 确保卷积层始终使用FP32精度
        self.gaussian_conv = self.gaussian_conv.float()
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()
    
    def _create_gaussian_kernel(self, size, sigma):
        """创建高斯核（FP32精度）"""
        # 使用更高效的向量化计算
        ax = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        # 高斯核公式计算
        sigma_sq = 2 * sigma ** 2
        kernel = torch.exp(-(xx**2 + yy**2) / sigma_sq) / (math.pi * sigma_sq)
        # 归一化
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]
    
    def forward(self, x):
        # 保存输入数据类型
        input_dtype = x.dtype
        # 高斯卷积必须在FP32精度下进行以确保数值稳定性
        with torch.cuda.amp.autocast(enabled=False):
            # 临时转换输入到FP32
            x_fp32 = x.to(torch.float32)
            edges_o = self.gaussian_conv(x_fp32)
            # 转换回原始精度进行后续操作
            edges_o = edges_o.to(input_dtype)
        # 批归一化和激活函数可以在混合精度下进行
        gaussian = self.act(self.norm(edges_o))
        return gaussian
    
    def half(self):
        """重写half方法以确保权重保持FP32精度"""
        super().half()
        # 确保高斯卷积权重保持FP32
        self.gaussian_conv.weight.data = self.gaussian_conv.weight.data.to(torch.float32)
        return self