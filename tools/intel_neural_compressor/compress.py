"""Intel Neural Compressor - MNIST Pruning Demo"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

# Model same as NNI SpatialNet for fair comparison
class SpatialNet4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.mean(x, dim=[2,3])
        return self.fc(x)

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

    # Train baseline
    print("Training baseline model...")
    model = SpatialNet4Layer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 3):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    baseline_acc = evaluate(model, test_loader)
    params = sum(p.numel() for p in model.parameters())
    print(f"Baseline: {baseline_acc:.2f}%, Params: {params:,}")

    # Prune with L1 unstructured 50%
    print("\nPruning with L1 unstructured 50%...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=0.5)

    pruned_acc = evaluate(model, test_loader)
    print(f"After pruning: {pruned_acc:.2f}%")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/mnist_inc_pruned.pt")

    # Quantize (PTQ)
    print("\nQuantizing to INT8 (PTQ)...")
    model_int8 = SpatialNet4Layer().to(device)
    model_int8.load_state_dict(torch.load("outputs/mnist_inc_pruned.pt", weights_only=False))
    model_int8.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_int8, inplace=True)
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 5: break
            model_int8(data)
    torch.quantization.convert(model_int8, inplace=True)
    quant_acc = evaluate(model_int8, test_loader)
    print(f"After quantization: {quant_acc:.2f}%")

    out = {
        "baseline": {"accuracy": round(baseline_acc, 2), "params": params},
        "pruned": {"accuracy": round(pruned_acc, 2), "method": "L1_unstructured_50%"},
        "quantized": {"accuracy": round(quant_acc, 2), "method": "PTQ_INT8"}
    }
    with open("outputs/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults: {out}")
    print("Saved to outputs/results.json")

if __name__ == "__main__":
    main()