"""PyTorch Native Compression - MNIST Pruning + Quantization Demo"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*7*7, 64); self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./test_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    print("Training baseline...")
    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 3):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    baseline_acc = evaluate(model, test_loader)
    params = sum(p.numel() for p in model.parameters())
    print(f"Baseline: {baseline_acc:.2f}%, Params: {params:,}")

    # L1 unstructured pruning 50%
    print("\nL1 unstructured pruning 50%...")
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, 'weight', 0.5)
    pruned_acc = evaluate(model, test_loader)
    print(f"After prune: {pruned_acc:.2f}%")

    # Dynamic quantization
    print("\nDynamic quantization to INT8...")
    quantized_model = MnistCNN().to(device)
    quantized_model.load_state_dict(model.state_dict())
    quantized_model = torch.quantization.quantize_dynamic(quantized_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
    quant_acc = evaluate(quantized_model, test_loader)
    print(f"After quant: {quant_acc:.2f}%")

    os.makedirs("outputs", exist_ok=True)
    torch.save({"baseline_acc": baseline_acc, "pruned_acc": pruned_acc, "quant_acc": quant_acc, "params": params}, "outputs/results.json")
    print(f"\nBaseline: {baseline_acc:.2f}% | Pruned: {pruned_acc:.2f}% | Quantized: {quant_acc:.2f}%")
    print("Saved to outputs/results.json")

if __name__ == "__main__":
    main()