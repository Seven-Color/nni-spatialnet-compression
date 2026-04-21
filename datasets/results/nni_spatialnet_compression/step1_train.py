# Step 1: 训练MNIST SpatialNet模型
# 运行: python step1_train.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 导入模型
from step1_spatialnet_mnist import SpatialNet4Layer

def main():
    print("=" * 60)
    print("Step 1: 训练MNIST SpatialNet模型")
    print("=" * 60)
    
    device = 'cpu'
    print(f"使用设备: {device}")
    
    # 加载MNIST数据
    print("\n加载MNIST数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(val_dataset)} 样本")
    
    # 创建模型
    print("\n创建4层SpatialNet模型...")
    model = SpatialNet4Layer(in_channels=1, num_classes=10).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"参数量: {param_count:,}")
    
    # 训练模型
    print("\n开始训练 (2 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 300 == 0:
                print(f'  Epoch [{epoch+1}/2] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/2] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%')
    
    # 评估模型
    print("\n评估模型...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'param_count': param_count
    }, 'mnist_spatialnet_trained.pth')
    
    print(f"\n模型已保存到: mnist_spatialnet_trained.pth")
    print("=" * 60)
    print("Step 1 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()