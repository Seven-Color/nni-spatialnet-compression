# Step 1: SpatialNet模型定义
# ============================================================
# SpatialNet - 一种用于空间数据的卷积神经网络
# 该模型包含典型的空间特征提取层，用于演示NNI的剪枝和量化功能
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 用于学习特征图的空间权重
    这是许多现代网络（如CBAM、SA-Net）的核心组件
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 生成空间注意力图
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class SpatialBlock(nn.Module):
    """
    空间块 - 包含卷积、归一化、激活和空间注意力的基本单元
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = SpatialAttention(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.attention(x)
        return x


class SpatialNet(nn.Module):
    """
    SpatialNet - 主网络结构
    
    一个用于空间特征提取的卷积神经网络，包含：
    - 初始卷积层进行特征提取
    - 多个SpatialBlock进行空间特征学习
    - 全局平均池化
    - 全连接层进行分类
    
    Args:
        in_channels: 输入通道数（默认3，用于RGB图像）
        num_classes: 分类类别数（默认10）
        width_multiplier: 宽度乘数，用于模型缩放
    """
    def __init__(self, in_channels=3, num_classes=10, width_multiplier=1.0):
        super(SpatialNet, self).__init__()
        
        # 计算各层通道数（应用宽度乘数）
        channels = [32, 64, 128, 256]
        channels = [int(c * width_multiplier) for c in channels]
        
        # 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # 空间块阶段
        self.stage1 = self._make_stage(channels[0], channels[1], num_blocks=2)
        self.stage2 = self._make_stage(channels[1], channels[2], num_blocks=2)
        self.stage3 = self._make_stage(channels[2], channels[3], num_blocks=2)
        
        # 下采样
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全局特征整合
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[3], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(SpatialBlock(in_channels, out_channels))
        for _ in range(num_blocks - 1):
            layers.append(SpatialBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始特征提取
        x = self.stem(x)
        
        # 空间块阶段1
        x = self.stage1(x)
        x = self.pool1(x)
        
        # 空间块阶段2
        x = self.stage2(x)
        x = self.pool2(x)
        
        # 空间块阶段3
        x = self.stage3(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x
    
    def get_parameter_count(self):
        """返回模型的可训练参数总数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_names(self):
        """返回所有可压缩的层名称"""
        layer_names = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    layer_names.append(name)
        return layer_names


def create_spatialnet_model(in_channels=3, num_classes=10, width_multiplier=1.0):
    """
    创建SpatialNet模型的工厂函数
    
    Args:
        in_channels: 输入通道数
        num_classes: 分类类别数
        width_multiplier: 宽度乘数
        
    Returns:
        SpatialNet模型实例
    """
    model = SpatialNet(in_channels=in_channels, num_classes=num_classes, 
                       width_multiplier=width_multiplier)
    return model


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("SpatialNet模型结构测试")
    print("=" * 60)
    
    # 创建模型
    model = create_spatialnet_model(in_channels=3, num_classes=10, width_multiplier=1.0)
    
    # 打印模型结构
    print(f"\n模型总参数量: {model.get_parameter_count():,}")
    print(f"\n层名称列表:")
    for name in model.get_layer_names():
        print(f"  - {name}")
    
    # 测试前向传播
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)
    
    # 创建虚拟输入
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    
    # 设置为评估模式
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出 logits:\n{output}")
    
    # 计算各层输出尺寸
    print("\n" + "=" * 60)
    print("各层输出尺寸")
    print("=" * 60)
    
    model.train()
    x = dummy_input
    layer_shapes = {}
    
    def hook_fn(module, input, output, name):
        layer_shapes[name] = output.shape
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            hooks.append(hook)
    
    with torch.no_grad():
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    for name, shape in layer_shapes.items():
        print(f"  {name}: {shape}")
    
    print("\n模型定义完成!")
