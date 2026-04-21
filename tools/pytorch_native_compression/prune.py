"""
MNIST CNN Pruning Script
Applies multiple PyTorch-native pruning methods and evaluates the model.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

# ── Model Definition (must match train.py) ────────────────────────────────────

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


def get_model_size(model, path="temp_model.pt"):
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    os.remove(path)
    return size_kb


def count_nonzero(model):
    return sum(p.numel() for p in model.parameters() if p is not None)


def evaluate(model, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

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
    """L1 unstructured pruning on all Conv2d and Linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)


def prune_model_random_unstructured(model, amount=0.3):
    """Random unstructured pruning."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.random_unstructured(module, name='weight', amount=amount)


def prune_model_ln(model, amount=0.4, n=2):
    """Ln unstructured pruning (default L2)."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.ln_structured(module, name='weight', amount=amount, n=n, dim=0)


def prune_model_structured(model, amount=0.3):
    """Random structured pruning — removes entire channels."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, name='weight', amount=amount, dim=0)


def remove_pruning_reparam(model):
    """Remove pruning re-parameterization so weights are permanently sparse."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')


def get_sparsity(model):
    """Return (sparse_count, total_count, sparsity_pct)."""
    total = 0
    sparse = 0
    for p in model.parameters():
        if p is not None:
            total += p.numel()
            sparse += (p == 0).sum().item()
    return sparse, total, 100.0 * sparse / total


def run_pruning_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prune] Using device: {device}")

    # Load baseline model
    model_path = "outputs/mnist_cnn.pt"
    if not os.path.exists(model_path):
        print("[prune] ERROR: trained model not found. Run train.py first.")
        return None

    baseline = MnistCNN().to(device)
    baseline.load_state_dict(torch.load(model_path, map_location=device))
    baseline_acc = evaluate(baseline, device)
    baseline_size = get_model_size(baseline, "outputs/baseline_temp.pt")
    baseline_params = sum(p.numel() for p in baseline.parameters())

    print(f"[prune] Baseline accuracy: {baseline_acc:.2f}%")
    print(f"[prune] Baseline size: {baseline_size:.2f} KB")

    results = []

    # ── Experiment 1: L1 Unstructured (50%) ─────────────────────────────────
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    prune_model_l1(model, amount=0.5)
    acc = evaluate(model, device)
    sparse, total, sparsity = get_sparsity(model)
    size = get_model_size(model, "outputs/pruned_l1_temp.pt")
    results.append({
        "method": "L1 Unstructured 50%",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "sparsity_pct": round(sparsity, 2),
        "sparse_weights": sparse,
        "total_weights": total,
    })
    print(f"[prune] L1 50%: Acc={acc:.2f}%, Sparsity={sparsity:.2f}%, Size={size:.2f} KB")

    # ── Experiment 2: L1 Unstructured (70%) ──────────────────────────────────
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    prune_model_l1(model, amount=0.7)
    acc = evaluate(model, device)
    sparse, total, sparsity = get_sparsity(model)
    size = get_model_size(model, "outputs/pruned_l1_70_temp.pt")
    results.append({
        "method": "L1 Unstructured 70%",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "sparsity_pct": round(sparsity, 2),
        "sparse_weights": sparse,
        "total_weights": total,
    })
    print(f"[prune] L1 70%: Acc={acc:.2f}%, Sparsity={sparsity:.2f}%, Size={size:.2f} KB")

    # ── Experiment 3: Random Unstructured (30%) ───────────────────────────────
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    prune_model_random_unstructured(model, amount=0.3)
    acc = evaluate(model, device)
    sparse, total, sparsity = get_sparsity(model)
    size = get_model_size(model, "outputs/pruned_random_temp.pt")
    results.append({
        "method": "Random Unstructured 30%",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "sparsity_pct": round(sparsity, 2),
        "sparse_weights": sparse,
        "total_weights": total,
    })
    print(f"[prune] Random 30%: Acc={acc:.2f}%, Sparsity={sparsity:.2f}%, Size={size:.2f} KB")

    # ── Experiment 4: Ln Structured (40%) ────────────────────────────────────
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    prune_model_ln(model, amount=0.4, n=2)
    acc = evaluate(model, device)
    sparse, total, sparsity = get_sparsity(model)
    size = get_model_size(model, "outputs/pruned_ln_temp.pt")
    results.append({
        "method": "Ln Structured 40% (dim=0)",
        "accuracy": round(acc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc, 2),
        "size_kb": round(size, 2),
        "sparsity_pct": round(sparsity, 2),
        "sparse_weights": sparse,
        "total_weights": total,
    })
    print(f"[prune] Ln Structured 40%: Acc={acc:.2f}%, Sparsity={sparsity:.2f}%, Size={size:.2f} KB")

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        "task": "pruning",
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_size_kb": round(baseline_size, 2),
        "baseline_params": baseline_params,
        "experiments": results,
    }
    with open("outputs/prune_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[prune] Results saved to outputs/prune_results.json")
    return out


if __name__ == "__main__":
    run_pruning_experiments()
