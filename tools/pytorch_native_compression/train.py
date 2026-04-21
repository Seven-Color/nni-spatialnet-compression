"""
MNIST CNN Training Script - Lightweight version
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Simple CNN
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28->14
        x = self.pool(self.relu(self.conv2(x)))  # 14->7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    device = torch.device("cpu")
    print(f"[train] Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 3
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)

        train_acc = 100.0 * correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total   += target.size(0)

        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/mnist_cnn.pt")

    params = sum(p.numel() for p in model.parameters())
    print(f"\n[train] Best accuracy: {best_acc:.2f}%, Params: {params}")

    out = {
        "task": "train",
        "best_test_acc": round(best_acc, 2),
        "params": params,
    }
    with open("outputs/train_results.json", "w") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    train()