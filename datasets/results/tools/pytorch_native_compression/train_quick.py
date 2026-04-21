"""Quick MNIST CNN Training - 2 epochs only"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def main():
    device = torch.device("cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./test_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Training 2 epochs...")
    for epoch in range(1, 3):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        train_acc = 100. * correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, pred = output.max(1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        test_acc = 100. * test_correct / test_total
        print(f"Epoch {epoch}: Train {train_acc:.2f}%, Test {test_acc:.2f}%")

    params = sum(p.numel() for p in model.parameters())
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/mnist_cnn.pt")
    print(f"Done! Params: {params:,}, Test Acc: {test_acc:.2f}%")

    out = {"params": params, "test_acc": round(test_acc, 2)}
    with open("outputs/train_results.json", "w") as f:
        json.dump(out, f)
    print(f"Saved to outputs/train_results.json")

if __name__ == "__main__":
    main()