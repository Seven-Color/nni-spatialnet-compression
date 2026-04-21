"""PyTorch Native Compression - Ultra Quick Version (1 epoch)"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16*7*7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./test_data', train=False, download=True, transform=transform), batch_size=512, shuffle=False)

model = TinyCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Training 1 epoch...")
model.train()
for data, target in train_loader:
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(data), target)
    loss.backward()
    optimizer.step()

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        _, pred = model(data).max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
baseline_acc = 100. * correct / total
params = sum(p.numel() for p in model.parameters())
print(f"Baseline: {baseline_acc:.2f}%, Params: {params:,}")

# Prune 50%
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(m, 'weight', 0.5)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        _, pred = model(data).max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
pruned_acc = 100. * correct / total
print(f"Pruned: {pruned_acc:.2f}%")

# Quantize
qmodel = TinyCNN()
qmodel.load_state_dict(model.state_dict())
qmodel = torch.quantization.quantize_dynamic(qmodel, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        _, pred = qmodel(data).max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
quant_acc = 100. * correct / total
print(f"Quantized: {quant_acc:.2f}%")

os.makedirs('outputs', exist_ok=True)
with open('outputs/results.json', 'w') as f:
    json.dump({'baseline': baseline_acc, 'pruned': pruned_acc, 'quantized': quant_acc, 'params': params}, f)
print("Done! Results saved.")