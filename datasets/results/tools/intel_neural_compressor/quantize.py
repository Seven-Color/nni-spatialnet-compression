"""
Quantization Script for MNIST CNN
Uses PyTorch dynamic quantization (int8) as fallback.
Note: Intel Neural Compressor is not available on Windows; falling back to PyTorch native.
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import MNISTNet


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cpu")
    script_dir = os.path.dirname(__file__)

    # Load baseline model
    model_path = os.path.join(script_dir, "mnist_cnn_trained.pt")
    if not os.path.exists(model_path):
        print("ERROR: mnist_cnn_trained.pt not found. Run train.py first.")
        return

    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Baseline accuracy
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_acc:.2f}%")

    # Dynamic quantization (int8)
    print("\nApplying dynamic quantization (int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

    # Evaluate quantized model
    quantized_acc = evaluate(quantized_model, test_loader, device)
    print(f"Quantized accuracy: {quantized_acc:.2f}%")

    # Save quantized model
    quantized_path = os.path.join(script_dir, "mnist_cnn_quantized.pt")
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Quantized model saved to: {quantized_path}")

    # Save results
    results = {
        "quantized": {
            "accuracy": round(quantized_acc, 2),
            "precision": "int8"
        }
    }
    results_path = os.path.join(script_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
