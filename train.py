"""训练模块"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, count_params


def load_data(config):
    """加载 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    data_cfg = config['dataset']
    train_ds = datasets.MNIST(
        data_cfg['root'], train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        data_cfg['root'], train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=config['train']['batch_size'], shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False
    )
    
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct, total = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct, total = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return 100. * correct / total


def train(config):
    """训练主函数"""
    print("\n" + "="*60)
    print("📦 训练阶段")
    print("="*60)
    
    device = torch.device("cpu")
    train_loader, test_loader = load_data(config)
    
    model = build_model(config).to(device)
    params = count_params(model)
    
    print(f"模型: {model.name}")
    print(f"参数量: {params:,}")
    print(f"训练 epochs: {config['train']['epochs']}")
    
    # 优化器
    train_cfg = config['train']
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_acc = 0
    for epoch in range(1, train_cfg['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch}/{train_cfg['epochs']} | "
              f"Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    # 保存模型和结果
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "baseline.pt")
    torch.save(model.state_dict(), model_path)
    
    result = {
        "accuracy": round(best_acc, 2),
        "params": params,
        "model_path": model_path,
        "model_name": model.name
    }
    
    result_path = os.path.join(output_dir, "train_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 训练完成! 测试准确率: {best_acc:.2f}%")
    print(f"   模型保存: {model_path}")
    
    return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    train(config)
