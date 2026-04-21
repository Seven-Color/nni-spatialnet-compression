"""
MNIST CNN Training Script
Train a 3-layer CNN on MNIST to achieve 98%+ accuracy.
Saves model to mnist_cnn_trained.pt
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Model ──────────────────────────────────────────────────────────────────────
class MNISTNet(nn.Module):
    """3-layer CNN for MNIST classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))  # ensure 3x3 output
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))   # 28x28 -> 28x28
        x = self.pool(self.relu(self.conv2(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv3(x)))  # 14x14 -> 7x7
        x = self.adaptive_pool(x)      # 7x7 -> 3x3
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total
        scheduler.step()
        print(f"Epoch {epoch+1:2d} | Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), "mnist_cnn_trained.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nBest accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")

    # Output results
    params = count_params(model)
    results = {
        "baseline": {
            "accuracy": round(best_acc, 2),
            "params": params
        }
    }
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    # Merge with existing results if any
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    return model, best_acc, params


if __name__ == "__main__":
    train()
