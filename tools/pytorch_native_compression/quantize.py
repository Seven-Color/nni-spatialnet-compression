"""
MNIST CNN Quantization Script
Applies PyTorch-native PTQ (Post-Training Quantization) to the trained model.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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


def get_model_size(path):
    size_kb = os.path.getsize(path) / 1024
    return size_kb


def get_size_of_state_dict(state_dict):
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(state_dict, tmp.name)
    size_kb = os.path.getsize(tmp.name) / 1024
    tmp.close()
    os.remove(tmp.name)
    return size_kb


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


def run_quantization_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quantize] Using device: {device}")

    model_path = "outputs/mnist_cnn.pt"
    if not os.path.exists(model_path):
        print("[quantize] ERROR: trained model not found. Run train.py first.")
        return None

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline = MnistCNN().to(device)
    baseline.load_state_dict(torch.load(model_path, map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    baseline_acc = evaluate(baseline, device, test_loader)
    baseline_size = get_size_of_state_dict(baseline.state_dict())
    print(f"[quantize] Baseline accuracy: {baseline_acc:.2f}%")
    print(f"[quantize] Baseline size: {baseline_size:.2f} KB")

    results = []

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 1: Dynamic Quantization (weights only, per-channel for Linear)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[quantize] === Dynamic Quantization (weights) ===")
    model_dq = MnistCNN().to(device)
    model_dq.load_state_dict(torch.load(model_path, map_location=device))

    model_dq.cpu()
    quantized_dq = torch.quantization.quantize_dynamic(
        model_dq,
        {nn.Linear},           # only quantize Linear layers
        dtype=torch.qint8
    )
    quantized_dq.to(device)

    acc_dq = evaluate(quantized_dq, device, test_loader)
    size_dq = get_size_of_state_dict(quantized_dq.state_dict())
    print(f"[quantize] Dynamic Quant: Acc={acc_dq:.2f}%, Size={size_dq:.2f} KB")
    results.append({
        "method": "Dynamic Quantization (Linear, int8)",
        "accuracy": round(acc_dq, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc_dq, 2),
        "size_kb": round(size_dq, 2),
        "size_reduction_pct": round(100 * (1 - size_dq / baseline_size), 2),
    })

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 2: Static Quantization (full post-training quantization)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[quantize] === Static Quantization (PTQ) ===")
    model_sq = MnistCNN().to(device)
    model_sq.load_state_dict(torch.load(model_path, map_location=device))
    model_sq.eval()

    # Fuse Conv+BN+ReLU patterns where possible
    # model_sq = torch.quantization.fuse_modules(model_sq, [['conv1', 'relu']])

    # Set qconfig for x86 (fbgemm) or CPU (qnnpack)
    backend = "fbgemm" if device.type == "cpu" else "qnnpack"
    model_sq.qconfig = torch.quantization.get_default_qconfig(backend)
    print(f"[quantize] Using qconfig backend: {backend}")
    print(f"[quantize] Model qconfig: {model_sq.qconfig}")

    # Prepare with calibration data
    torch.quantization.prepare(model_sq, inplace=True)

    # Calibrate with a subset of test data
    calib_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)
    model_sq.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(calib_loader):
            if i >= 8:  # ~512 samples for calibration
                break
            model_sq(data)

    quantized_sq = torch.quantization.convert(model_sq, inplace=False)
    acc_sq = evaluate(quantized_sq, device, test_loader)
    size_sq = get_size_of_state_dict(quantized_sq.state_dict())
    print(f"[quantize] Static Quant: Acc={acc_sq:.2f}%, Size={size_sq:.2f} KB")
    results.append({
        "method": "Static Quantization (PTQ, int8, per-tensor)",
        "accuracy": round(acc_sq, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc_sq, 2),
        "size_kb": round(size_sq, 2),
        "size_reduction_pct": round(100 * (1 - size_sq / baseline_size), 2),
    })

    # ══════════════════════════════════════════════════════════════════════════
    # Experiment 3: Static Quantization with per-channel quantization
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[quantize] === Static Quantization (per-channel) ===")
    model_sq_pc = MnistCNN().to(device)
    model_sq_pc.load_state_dict(torch.load(model_path, map_location=device))
    model_sq_pc.eval()

    # Use per-channel quantization for better accuracy
    model_sq_pc.qconfig = torch.quantization.get_default_per_channel_qconfig(backend)
    print(f"[quantize] Per-channel qconfig: {model_sq_pc.qconfig}")

    torch.quantization.prepare(model_sq_pc, inplace=True)

    with torch.no_grad():
        for i, (data, target) in enumerate(calib_loader):
            if i >= 8:
                break
            model_sq_pc(data)

    quantized_sq_pc = torch.quantization.convert(model_sq_pc, inplace=False)
    acc_sq_pc = evaluate(quantized_sq_pc, device, test_loader)
    size_sq_pc = get_size_of_state_dict(quantized_sq_pc.state_dict())
    print(f"[quantize] Static Quant (per-channel): Acc={acc_sq_pc:.2f}%, Size={size_sq_pc:.2f} KB")
    results.append({
        "method": "Static Quantization (PTQ, int8, per-channel)",
        "accuracy": round(acc_sq_pc, 2),
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy_drop": round(baseline_acc - acc_sq_pc, 2),
        "size_kb": round(size_sq_pc, 2),
        "size_reduction_pct": round(100 * (1 - size_sq_pc / baseline_size), 2),
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "task": "quantization",
        "baseline_accuracy": round(baseline_acc, 2),
        "baseline_size_kb": round(baseline_size, 2),
        "experiments": results,
    }
    with open("outputs/quantize_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[quantize] Results saved to outputs/quantize_results.json")
    return out


if __name__ == "__main__":
    run_quantization_experiments()
