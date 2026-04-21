# Step 2: 数据准备与训练设置
# ============================================================
# 本步骤创建模拟数据集和训练循环，用于后续的剪枝和量化
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Callable, Any
import nni


class DummySpatialDataset(Dataset):
    """
    模拟的空间数据数据集
    用于演示目的，生成随机但可重复的空间数据
    """
    def __init__(self, num_samples=1000, image_size=32, in_channels=3, num_classes=10, 
                 train=True, seed=42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.train = train
        
        # 设置随机种子以确保可重复性
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 预生成所有数据
        self.images = torch.randn(num_samples, in_channels, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # 添加一些结构化特征使数据更真实
        for i in range(num_samples):
            label = self.labels[i].item()
            # 为每个类别添加不同的空间模式
            pattern = self._generate_spatial_pattern(label, image_size)
            self.images[i] = self.images[i] * 0.3 + pattern * 0.7
    
    def _generate_spatial_pattern(self, label, size):
        """为每个类别生成独特的空间模式"""
        x = torch.linspace(-1, 1, size)
        y = torch.linspace(-1, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # 基于label选择不同的模式
        patterns = [
            torch.exp(-(xx**2 + yy**2) / 0.5),  # 高斯斑
            torch.abs(xx) + torch.abs(yy),       # 菱形
            torch.sin(xx * np.pi * (label + 1)), # 正弦波
            torch.cos(yy * np.pi * (label + 1)), # 余弦波
            (xx > 0).float() * (yy > 0).float(), # 象限
            torch.sin(xx * yy * np.pi * 2),      #干涉图
            torch.max(torch.abs(xx), torch.abs(yy)), # 正方形
            torch.atan2(yy, xx),                 # 角度场
        ]
        
        pattern_idx = label % len(patterns)
        pattern = patterns[pattern_idx]
        
        return pattern.unsqueeze(0).repeat(3, 1, 1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_data_loaders(batch_size=64, num_samples=1000, image_size=32,
                         in_channels=3, num_classes=10):
    """
    创建训练和验证数据加载器
    
    Args:
        batch_size: 批次大小
        num_samples: 样本总数
        image_size: 图像尺寸
        in_channels: 输入通道数
        num_classes: 类别数
        
    Returns:
        (train_loader, val_loader) 元组
    """
    # 分割训练集和验证集
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset = DummySpatialDataset(
        num_samples=train_size, 
        image_size=image_size, 
        in_channels=in_channels, 
        num_classes=num_classes,
        train=True,
        seed=42
    )
    
    val_dataset = DummySpatialDataset(
        num_samples=val_size, 
        image_size=image_size, 
        in_channels=in_channels, 
        num_classes=num_classes,
        train=False,
        seed=123
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device = None) -> float:
    """
    训练一个epoch
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        平均训练损失
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_model(model: nn.Module, val_loader: DataLoader,
                   device: torch.device = None) -> Tuple[float, float]:
    """
    评估模型
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        device: 计算设备
        
    Returns:
        (损失, 准确率) 元组
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def training_step(batch, model, criterion=F.cross_entropy):
    """
    NNI压缩使用的训练步骤函数
    
    Args:
        batch: 包含(images, labels)的元组
        model: 模型
        criterion: 损失函数
        
    Returns:
        损失值
    """
    images, labels = batch
    outputs = model(images)
    loss = criterion(outputs, labels)
    return loss


def full_training_loop(model: nn.Module, train_loader: DataLoader, 
                       val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                       max_epochs: int = 10, device: torch.device = None,
                       verbose: bool = True) -> dict:
    """
    完整的训练循环，用于基线训练和压缩后微调
    
    Args:
        model: 待训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        max_epochs: 最大训练轮数
        device: 计算设备
        verbose: 是否打印训练过程
        
    Returns:
        包含训练历史信息的字典
    """
    if device is None:
        device = next(model.parameters()).device
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        
        # 验证
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        if verbose and (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{max_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
    
    return history


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("数据准备与训练设置测试")
    print("=" * 60)
    
    # 创建数据加载器
    print("\n创建数据集...")
    train_loader, val_loader = create_data_loaders(
        batch_size=64, 
        num_samples=1000, 
        image_size=32,
        in_channels=3, 
        num_classes=10
    )
    
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    
    # 测试数据加载
    print("\n测试数据加载...")
    images, labels = next(iter(train_loader))
    print(f"批次图像形状: {images.shape}")
    print(f"批次标签形状: {labels.shape}")
    
    # 导入SpatialNet模型
    from step1_spatialnet_model import create_spatialnet_model
    
    # 创建模型
    print("\n创建模型...")
    model = create_spatialnet_model(in_channels=3, num_classes=10, width_multiplier=1.0)
    print(f"模型参数量: {model.get_parameter_count():,}")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试训练一个epoch
    print("\n" + "=" * 60)
    print("测试训练一个epoch")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    
    elapsed = time.time() - start_time
    
    print(f"\n训练损失: {train_loss:.4f}")
    print(f"验证损失: {val_loss:.4f}")
    print(f"验证准确率: {val_acc:.2f}%")
    print(f"耗时: {elapsed:.2f}秒")
    
    print("\n数据准备与训练设置完成!")
