"""快速验证脚本 - 直接运行，无需 conda"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    print("=" * 50)
    print("模型压缩快速验证 - 2层CNN")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("加载数据...")
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    channels = [16, 32]
    model = CNN2Layer(channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"模型: 2层CNN, 通道={channels}, 参数量={params:,}")
    
    # 训练
    print("\n训练 (2 epochs)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 3):
        model.train()
        total, correct = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        train_acc = 100. * correct / total
        print(f"  Epoch {epoch}: Train {train_acc:.2f}%")
    
    # 评估基线
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    baseline_acc = 100. * correct / total
    print(f"\n基线准确率: {baseline_acc:.2f}%")
    
    # 保存基线
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/baseline.pt")
    
    # 剪枝
    print("\n剪枝 (L1 Unstructured 50%)...")
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, 'weight', 0.5)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    pruned_acc = 100. * correct / total
    print(f"剪枝后准确率: {pruned_acc:.2f}%")
    
    # 量化
    print("\n量化 (Dynamic INT8)...")
    qmodel = CNN2Layer(channels).to(device)
    qmodel.load_state_dict(torch.load("results/baseline.pt", weights_only=False))
    qmodel = torch.quantization.quantize_dynamic(qmodel, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    
    qmodel.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = qmodel(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    quant_acc = 100. * correct / total
    print(f"量化后准确率: {quant_acc:.2f}%")
    
    # 结果汇总
    print("\n" + "=" * 50)
    print("📊 实验结果汇总")
    print("=" * 50)
    print(f"| 阶段   | 准确率   |")
    print(f"|--------|----------|")
    print(f"| 基线   | {baseline_acc:.2f}%   |")
    print(f"| 剪枝   | {pruned_acc:.2f}%   |")
    print(f"| 量化   | {quant_acc:.2f}%   |")
    
    out = {
        "baseline": {"accuracy": round(baseline_acc, 2), "params": params},
        "pruned": {"accuracy": round(pruned_acc, 2), "method": "L1_unstructured_50%"},
        "quantized": {"accuracy": round(quant_acc, 2), "method": "Dynamic_INT8"}
    }
    with open("results/quick_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ 结果已保存到 results/quick_results.json")

if __name__ == "__main__":
    main()
