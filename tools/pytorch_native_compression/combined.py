"""
MNIST CNN Combined Pruning + Quantization Script
Applies pruning first, then quantization to a pruned model.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

# ── Model Definition ──────────────────────────────────────────────────────────

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_size_of_state_dict(state_dict):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(state_dict, tmp.name)
    size_kb = os.path.getsize(tmp.name) / 1024
    tmp.close()
    os.remove(tmp.name)
    return size_kb


def get_sparsity(model):
    total = 0
    sparse = 0
    for p in model.parameters():
        if p is not None:
            total += p.numel()
            sparse += (p == 0).sum().item()
    return sparse, total, 100.0 * sparse / total


def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    return 100.0 * correct / total


def prune_model_l1(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)


def remove_pruning_reparam(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except Exception:
                pass


def run_combined_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[combined] Using device: {device}")

    model_path = "outputs/mnist_cnn.pt"
    if not os.path.exists(model_path):
        print("[combined] ERROR: trained model not found. Run train.py first.")
        return None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    calib_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline = MnistCNN().to(device)
    baseline.load_state_dict(torch.load(model_path, map_location=device))
    baseline_acc = evaluate(baseline, device, test_loader)
    baseline_size = get_size_of_state_dict(baseline.state_dict())
    print(f"[combined] Baseline accuracy: {baseline_acc:.2f}%")
    print(f"[combined] Baseline size: {baseline_size:.2f} KB")

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 1: Prune (L1 50%) → Quantize (Dynamic)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[combined] === Prune (L1 50%) + Dynamic Quantization ===")
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    prune_model_l1(model, amount=0.5)
    remove_pruning_reparam(model)

    sparse, total, sparsity = get_sparsity(model)
    print(f"[combined] After pruning: sparsity={sparsity:.2f}%")

    model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    quantized.to(device)

    acc = evaluate(quantized, device, test_loader)
    size = get_size_of_state_dict(quantized.state_dict())
    print(f"[combined] Pruned+DQ: Acc={acc:.2f}%, Size={size:.2f} KB, Sparsity={sparsity:.2f}%")
    results.append({
        "method": "Prune L1 50% + Dynamic Quant (Linear)",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "size_reduction_pct": round(100 * (1 - size / baseline_size), 2),
        "sparsity_pct": round(sparsity, 2),
    })

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 2: Prune (L1 70%) → Quantize (Dynamic)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[combined] === Prune (L1 70%) + Dynamic Quantization ===")
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    prune_model_l1(model, amount=0.7)
    remove_pruning_reparam(model)
    sparse, total, sparsity = get_sparsity(model)

    model.cpu()
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized.to(device)

    acc = evaluate(quantized, device, test_loader)
    size = get_size_of_state_dict(quantized.state_dict())
    print(f"[combined] Pruned+DQ: Acc={acc:.2f}%, Size={size:.2f} KB, Sparsity={sparsity:.2f}%")
    results.append({
        "method": "Prune L1 70% + Dynamic Quant (Linear)",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "size_reduction_pct": round(100 * (1 - size / baseline_size), 2),
        "sparsity_pct": round(sparsity, 2),
    })

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 3: Prune (L1 50%) → Quantize (Static PTQ)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[combined] === Prune (L1 50%) + Static Quantization (PTQ) ===")
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    prune_model_l1(model, amount=0.5)
    remove_pruning_reparam(model)
    sparse, total, sparsity = get_sparsity(model)
    model.eval()

    backend = "fbgemm"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (data, target) in enumerate(calib_loader):
            if i >= 8:
                break
            model(data)

    quantized = torch.quantization.convert(model, inplace=False)
    acc = evaluate(quantized, device, test_loader)
    size = get_size_of_state_dict(quantized.state_dict())
    print(f"[combined] Pruned+SQ: Acc={acc:.2f}%, Size={size:.2f} KB, Sparsity={sparsity:.2f}%")
    results.append({
        "method": "Prune L1 50% + Static Quant (PTQ)",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "size_reduction_pct": round(100 * (1 - size / baseline_size), 2),
        "sparsity_pct": round(sparsity, 2),
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "task": "combined_pruning_quantization",
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_size_kb": round(baseline_size, 2),
        "experiments": results,
    }
    with open("outputs/combined_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[combined] Results saved to outputs/combined_results.json")
    return out


if __name__ == "__main__":
    run_combined_experiments()
