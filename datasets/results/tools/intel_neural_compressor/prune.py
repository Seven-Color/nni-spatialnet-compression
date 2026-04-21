"""
Pruning Script for MNIST CNN
Uses PyTorch native torch.nn.utils.prune (L1 unstructured pruning at 50% sparsity).
Note: Intel Neural Compressor is not available on Windows; falling back to native PyTorch.
"""
import os
import json
import torch
import torch.nn as nn
from torch.nn.utils import prune
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


def compute_sparsity(model):
    total = 0
    zero = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
            zero += (p == 0).sum().item()
    return zero / total if total > 0 else 0.0


def prune_model(model, sparsity=0.5):
    """Apply L1 unstructured pruning to all conv and linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            # Make pruning permanent
            prune.remove(module, "weight")
    return model


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

    # Apply pruning
    print("\nApplying L1 unstructured pruning (50% sparsity)...")
    model = prune_model(model, sparsity=0.5)
    sparsity = compute_sparsity(model)
    print(f"Actual model sparsity: {sparsity:.4f}")

    # Evaluate pruned model
    pruned_acc = evaluate(model, test_loader, device)
    print(f"Pruned accuracy: {pruned_acc:.2f}%")

    # Save pruned model
    pruned_path = os.path.join(script_dir, "mnist_cnn_pruned.pt")
    torch.save(model.state_dict(), pruned_path)
    print(f"Pruned model saved to: {pruned_path}")

    # Save results
    results = {
        "pruned": {
            "accuracy": round(pruned_acc, 2),
            "sparsity": round(sparsity, 4)
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
