# 快速MNIST训练测试
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms

print("加载数据...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform), batch_size=128, shuffle=False)

print("创建模型...")
from step1_spatialnet_mnist import SpatialNet4Layer
model = SpatialNet4Layer(in_channels=1, num_classes=10)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

print("训练模型 (1 epoch)...")
opt = torch.optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    opt.zero_grad()
    crit(model(data), target).backward()
    opt.step()
    if (batch_idx + 1) % 300 == 0: print(f'  Batch {batch_idx+1}/{len(train_loader)}')

print("评估模型...")
model.eval()
correct = total = 0
with torch.no_grad():
    for data, target in val_loader:
        _, predicted = model(data).max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
print(f"测试集准确率: {100.*correct/total:.2f}%")
print("完成!")