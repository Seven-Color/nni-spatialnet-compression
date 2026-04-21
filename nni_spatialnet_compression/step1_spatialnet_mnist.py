# Step 1: MNIST分类任务 - 4层SpatialNet
# ============================================================
# 使用MNIST数据集训练4层SpatialNet进行手写数字分类
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os


# ============================================================
# 模型定义
# ============================================================

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class SpatialBlock(nn.Module):
    """SpatialBlock - Conv + BN + ReLU + Attention"""
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


class SpatialNet4Layer(nn.Module):
    """
    4层SpatialNet - 用于MNIST分类
    
    结构:
    - Layer1: Conv + BN + ReLU + Attention + Pool (1 -> 32)
    - Layer2: Conv + BN + ReLU + Attention + Pool (32 -> 64)
    - Layer3: Conv + BN + ReLU + Attention + Pool (64 -> 128)
    - Layer4: Conv + BN + ReLU + Attention (128 -> 256)
    - Classifier: GlobalAvgPool + Flatten + Linear (256 -> 10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(SpatialNet4Layer, self).__init__()
        
        # Layer 1
        self.layer1 = SpatialBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2
        self.layer2 = SpatialBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3
        self.layer3 = SpatialBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 4
        self.layer4 = SpatialBlock(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.pool2(x)
        
        x = self.layer3(x)
        x = self.pool3(x)
        
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        """返回softmax概率"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_names(self):
        layer_names = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    layer_names.append(name)
        return layer_names


def get_mnist_dataloaders(batch_size=64):
    """获取MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])
    
    # 下载并加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, epochs=2, lr=0.001, device='cpu'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f'  Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%')
    
    return model


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')
    return accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("MNIST 4层SpatialNet分类任务")
    print("=" * 60)
    
    # 获取MNIST数据加载器
    print("\n加载MNIST数据集...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建4层SpatialNet模型...")
    model = SpatialNet4Layer(in_channels=1, num_classes=10)
    print(f"模型参数量: {model.get_parameter_count():,}")
    
    # 测试前向传播
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)
    
    dummy_input = torch.randn(4, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出logits:\n{output}")
    
    # 训练模型
    print("\n" + "=" * 60)
    print("训练模型 (2 epochs)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = train_model(model, train_loader, epochs=2, lr=0.001, device=device)
    
    # 评估模型
    print("\n" + "=" * 60)
    print("评估模型")
    print("=" * 60)
    
    accuracy = evaluate_model(model, test_loader, device=device)
    
    # 测试预测
    print("\n" + "=" * 60)
    print("测试预测示例")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 1, 28, 28)
        prob = model.predict(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"预测概率: {prob}")
        print(f"预测数字: {prob.argmax(dim=1).item()}")
    
    print("\n训练和评估完成!")
