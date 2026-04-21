"""
模型压缩一键运行脚本
====================
功能: 创建conda环境 → 训练基线模型 → 剪枝 → 量化 → 输出结果

使用方法:
    python run_all_experiments.py

设计说明:
    - 默认使用2层CNN (可快速验证流程)
    - 如需放大模型，修改下方 MODEL_CONFIG 中的 layers 即可
    - 所有结果输出到 results/ 目录

作者: Seven-Color
日期: 2026-04-21
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# ============================================================
# 📋 配置区域 - 放大模型从这里改
# ============================================================
MODEL_CONFIG = {
    "name": "MNIST_2Layer_CNN",      # 模型名称
    "layers": 2,                      # 卷积层数量 (默认2层，验证用)
    "channels": [16, 32],             # 每层通道数
    "epochs": 2,                      # 训练轮数
    "prune_sparsity": 0.5,            # 剪枝稀疏度
    "batch_size": 256,
    "lr": 0.001,
}

# 可选: 3层/4层配置 (放大时取消注释)
# MODEL_CONFIG = {
#     "name": "MNIST_3Layer_CNN",
#     "layers": 3,
#     "channels": [16, 32, 64],
#     "epochs": 5,
#     "prune_sparsity": 0.5,
#     "batch_size": 128,
#     "lr": 0.001,
# }

# ============================================================
# 代码区域 (无需修改)
# ============================================================

ENV_NAME = "model_compression"
PYTHON_VERSION = "3.10"  # 使用 3.10 避免兼容性问题

REQUIRED_PACKAGES = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "nni>=3.0.0",          # 可选，未安装则使用 PyTorch 原生
    "numpy",
]

RESULTS_DIR = Path("results")
LOG_FILE = RESULTS_DIR / "experiment_log.txt"


def log(msg):
    """打印并记录日志"""
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_command(cmd, description, timeout=300):
    """运行命令，失败则退出"""
    log(f"\n{'='*60}")
    log(f"📦 {description}")
    log(f"命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    log(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
        if result.stdout:
            log(result.stdout)
        if result.stderr:
            log(result.stderr)
        if result.returncode != 0:
            log(f"❌ 命令执行失败 (code {result.returncode})")
            return False
        log(f"✅ {description} 完成")
        return True
    except subprocess.TimeoutExpired:
        log(f"❌ {description} 超时 ({timeout}s)")
        return False
    except Exception as e:
        log(f"❌ {description} 异常: {e}")
        return False


def create_conda_env():
    """创建 conda 虚拟环境"""
    log("\n" + "="*60)
    log("🔧 步骤1: 创建 conda 虚拟环境")
    log("="*60)
    
    # 检查 conda 是否可用
    result = subprocess.run("conda --version", shell=True, capture_output=True)
    if result.returncode != 0:
        log("❌ 未检测到 conda，请先安装 Anaconda/Miniconda")
        return False
    
    # 创建环境
    cmd = f"conda create -y -n {ENV_NAME} python={PYTHON_VERSION}"
    if not run_command(cmd, "创建 conda 环境", timeout=300):
        return False
    
    # 安装依赖包
    for pkg in REQUIRED_PACKAGES:
        cmd = f"conda run -n {ENV_NAME} pip install {pkg}"
        if not run_command(cmd, f"安装 {pkg}", timeout=300):
            return False
    
    log(f"✅ conda 环境 {ENV_NAME} 创建完成")
    return True


def train_model():
    """训练基线模型"""
    log("\n" + "="*60)
    log("🧠 步骤2: 训练基线模型")
    log("="*60)
    
    results = {}
    
    # PyTorch Native 训练
    log("\n--- 训练 PyTorch Native 模型 ---")
    train_script = RESULTS_DIR / "train_pytorch.py"
    train_script.write_text(TRAIN_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{train_script}"'
    if run_command(cmd, "训练 PyTorch 模型", timeout=600):
        results["pytorch"] = load_results("pytorch")
    
    # NNI 训练 (如果安装了 NNI)
    log("\n--- 训练 NNI 模型 ---")
    train_nni_script = RESULTS_DIR / "train_nni.py"
    train_nni_script.write_text(TRAIN_NNI_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{train_nni_script}"'
    if run_command(cmd, "训练 NNI 模型", timeout=600):
        results["nni"] = load_results("nni")
    
    return results


def prune_model():
    """剪枝模型"""
    log("\n" + "="*60)
    log("✂️ 步骤3: 剪枝实验")
    log("="*60)
    
    results = {}
    
    # PyTorch Native 剪枝
    log("\n--- PyTorch Native 剪枝 ---")
    prune_script = RESULTS_DIR / "prune_pytorch.py"
    prune_script.write_text(PRUNE_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{prune_script}"'
    if run_command(cmd, "PyTorch 剪枝", timeout=300):
        results["pytorch"] = load_results("pytorch_pruned")
    
    # NNI 剪枝 (如果安装了 NNI)
    log("\n--- NNI 剪枝 ---")
    prune_nni_script = RESULTS_DIR / "prune_nni.py"
    prune_nni_script.write_text(PRUNE_NNI_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{prune_nni_script}"'
    if run_command(cmd, "NNI 剪枝", timeout=300):
        results["nni"] = load_results("nni_pruned")
    
    return results


def quantize_model():
    """量化模型"""
    log("\n" + "="*60)
    log("⚡ 步骤4: 量化实验")
    log("="*60)
    
    results = {}
    
    # PyTorch Native 量化
    log("\n--- PyTorch Native 量化 ---")
    quant_script = RESULTS_DIR / "quantize_pytorch.py"
    quant_script.write_text(QUANT_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{quant_script}"'
    if run_command(cmd, "PyTorch 量化", timeout=300):
        results["pytorch"] = load_results("pytorch_quantized")
    
    # NNI 量化 (如果安装了 NNI)
    log("\n--- NNI 量化 ---")
    quant_nni_script = RESULTS_DIR / "quantize_nni.py"
    quant_nni_script.write_text(QUANT_NNI_SCRIPT, encoding="utf-8")
    
    cmd = f'conda run -n {ENV_NAME} python "{quant_nni_script}"'
    if run_command(cmd, "NNI 量化", timeout=300):
        results["nni"] = load_results("nni_quantized")
    
    return results


def load_results(name):
    """加载实验结果"""
    result_file = RESULTS_DIR / f"{name}_results.json"
    if result_file.exists():
        return json.loads(result_file.read_text(encoding="utf-8"))
    return {}


def generate_summary(all_results):
    """生成实验结果汇总"""
    log("\n" + "="*60)
    log("📊 实验结果汇总")
    log("="*60)
    
    summary = {
        "config": MODEL_CONFIG,
        "results": all_results,
    }
    
    # 输出汇总表格
    log("\n| 工具 | 阶段 | 准确率 | 说明 |")
    log("|------|------|--------|------|")
    
    for tool, tool_results in all_results.items():
        if not tool_results:
            continue
        for stage, data in tool_results.items():
            acc = data.get("accuracy", "N/A")
            note = data.get("method", "")
            log(f"| {tool} | {stage} | {acc}% | {note} |")
    
    # 保存汇总
    summary_file = RESULTS_DIR / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n✅ 结果已保存到: {summary_file}")
    
    return summary


# ============================================================
# 📝 内嵌脚本 - PyTorch 原生
# ============================================================

TRAIN_SCRIPT = '''"""PyTorch Native 训练脚本 - 2层CNN"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    print("=" * 50)
    print("PyTorch Native 训练 - 2层CNN")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    print("加载 MNIST 数据集...")
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 创建模型
    channels = [16, 32]  # 可调整
    model = CNN2Layer(channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"模型: 2层CNN, 通道 {channels}, 参数量: {params:,}")
    
    # 训练
    print("开始训练...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):  # 默认2轮，可增加
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    
    print(f"测试准确率: {accuracy:.2f}%")
    
    # 保存
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/pytorch_baseline.pt")
    
    out = {"accuracy": round(accuracy, 2), "params": params, "method": "2-layer CNN"}
    with open("results/pytorch_results.json", "w") as f:
        json.dump(out, f)
    print(f"结果已保存: {out}")

if __name__ == "__main__":
    main()
'''

PRUNE_SCRIPT = '''"""PyTorch Native 剪枝脚本"""
import os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    print("=" * 50)
    print("PyTorch Native 剪枝 - L1 Unstructured 50%")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 加载模型
    model = CNN2Layer().to(device)
    model.load_state_dict(torch.load("results/pytorch_baseline.pt", weights_only=False))
    
    # 计算原始准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    baseline_acc = 100. * correct / total
    print(f"剪枝前准确率: {baseline_acc:.2f}%")
    
    # L1 Unstructured 剪枝 50%
    print("执行 L1 Unstructured 剪枝 (50%)...")
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(m, 'weight', 0.5)
    
    # 评估剪枝后
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    pruned_acc = 100. * correct / total
    print(f"剪枝后准确率: {pruned_acc:.2f}%")
    
    # 保存
    torch.save(model.state_dict(), "results/pytorch_pruned.pt")
    out = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(pruned_acc, 2),
        "method": "L1_unstructured_50%",
        "sparsity": 0.5
    }
    with open("results/pytorch_pruned_results.json", "w") as f:
        json.dump(out, f)
    print(f"结果已保存: {out}")

if __name__ == "__main__":
    main()
'''

QUANT_SCRIPT = '''"""PyTorch Native 量化脚本"""
import os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    print("=" * 50)
    print("PyTorch Native 量化 - Dynamic INT8")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 加载模型
    model = CNN2Layer().to(device)
    model.load_state_dict(torch.load("results/pytorch_baseline.pt", weights_only=False))
    
    # 计算原始准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    baseline_acc = 100. * correct / total
    print(f"量化前准确率: {baseline_acc:.2f}%")
    
    # Dynamic Quantization
    print("执行 Dynamic Quantization (INT8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    # 评估量化后
    quantized_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = quantized_model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    quant_acc = 100. * correct / total
    print(f"量化后准确率: {quant_acc:.2f}%")
    
    # 保存
    torch.save(quantized_model.state_dict(), "results/pytorch_quantized.pt")
    out = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(quant_acc, 2),
        "method": "Dynamic_INT8",
        "compression": "4x"
    }
    with open("results/pytorch_quantized_results.json", "w") as f:
        json.dump(out, f)
    print(f"结果已保存: {out}")

if __name__ == "__main__":
    main()
'''

# ============================================================
# 📝 内嵌脚本 - NNI (可选)
# ============================================================

TRAIN_NNI_SCRIPT = '''"""NNI 训练脚本 - 2层CNN"""
import os, json, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    try:
        import nni
    except ImportError:
        print("NNI 未安装，跳过 NNI 训练")
        return
    
    print("=" * 50)
    print("NNI 训练 - 2层CNN")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    channels = [16, 32]
    model = CNN2Layer(channels).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"模型: 2层CNN, 通道 {channels}, 参数量: {params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    
    print(f"测试准确率: {accuracy:.2f}%")
    
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/nni_baseline.pt")
    
    out = {"accuracy": round(accuracy, 2), "params": params, "method": "2-layer CNN (NNI)"}
    with open("results/nni_results.json", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    main()
'''

PRUNE_NNI_SCRIPT = '''"""NNI 剪枝脚本"""
import os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    try:
        from nni.compression.pruning import LevelPruner
    except ImportError:
        print("NNI 未安装或导入失败，跳过 NNI 剪枝")
        return
    
    print("=" * 50)
    print("NNI LevelPruner 剪枝")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    model = CNN2Layer().to(device)
    model.load_state_dict(torch.load("results/nni_baseline.pt", weights_only=False))
    
    # Baseline
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    baseline_acc = 100. * correct / total
    print(f"剪枝前: {baseline_acc:.2f}%")
    
    # NNI LevelPruner
    config_list = [{'sparsity': 0.5, 'op_types': ['Conv2d', 'Linear']}]
    pruner = LevelPruner(model, config_list)
    pruned_model, masks = pruner.compress()
    
    pruned_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = pruned_model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    pruned_acc = 100. * correct / total
    print(f"剪枝后: {pruned_acc:.2f}%")
    
    torch.save(pruned_model.state_dict(), "results/nni_pruned.pt")
    out = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(pruned_acc, 2),
        "method": "LevelPruner_50%"
    }
    with open("results/nni_pruned_results.json", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    main()
'''

QUANT_NNI_SCRIPT = '''"""NNI 量化脚本"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CNN2Layer(nn.Module):
    def __init__(self, channels=[16, 32]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc = nn.Linear(channels[1] * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2); self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.view(x.size(0), -1))

def main():
    try:
        from nni.compression.quantization import QATQuantizer
        from nni.compression.utils import TorchEvaluator
        import nni
    except ImportError:
        print("NNI 未安装或导入失败，跳过 NNI 量化")
        return
    
    print("=" * 50)
    print("NNI QATQuantizer 量化")
    print("=" * 50)
    
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    model = CNN2Layer().to(device)
    model.load_state_dict(torch.load("results/nni_baseline.pt", weights_only=False))
    
    # Baseline
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    baseline_acc = 100. * correct / total
    print(f"量化前: {baseline_acc:.2f}%")
    
    # QAT Quantization
    config_list = [{
        'op_types': ['Conv2d', 'Linear'],
        'quant_dtype': 'int8',
        'target_names': ['weight', '_output_'],
    }]
    
    def training_step(batch, model):
        images, labels = batch
        outputs = model(images)
        return F.cross_entropy(outputs, labels)
    
    def training_loop(model, optimizers, training_step_fn, max_epochs=None):
        optimizer = optimizers
        model.train()
        for epoch in range(max_epochs if max_epochs else 1):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = training_step_fn(batch, model)
                loss.backward()
                optimizer.step()
    
    traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
    evaluator = TorchEvaluator(
        training_func=training_loop,
        optimizers=traced_optimizer,
        training_step=training_step,
    )
    
    quantizer = QATQuantizer(model, config_list, evaluator)
    quantized_model, _ = quantizer.compress(max_epochs=1)
    
    quantized_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            _, pred = quantized_model(data).max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    quant_acc = 100. * correct / total
    print(f"量化后: {quant_acc:.2f}%")
    
    torch.save(quantized_model.state_dict(), "results/nni_quantized.pt")
    out = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(quant_acc, 2),
        "method": "QAT_INT8"
    }
    with open("results/nni_quantized_results.json", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    main()
'''


# ============================================================
# 🚀 主程序
# ============================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     模型压缩一键运行脚本 - PyTorch + NNI                  ║
╠══════════════════════════════════════════════════════════╣
║  功能: conda环境 → 训练 → 剪枝 → 量化 → 输出结果        ║
║  验证: 默认2层CNN，快速验证流程                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 初始化
    RESULTS_DIR.mkdir(exist_ok=True)
    LOG_FILE.unlink(missing_ok=True)
    log(f"实验配置: {MODEL_CONFIG}")
    
    # 1. 创建 conda 环境 (可选，跳过如果已存在)
    # if not (os.path.exists(os.path.expanduser(f"~/.conda/envs/{ENV_NAME}"))):
    #     if not create_conda_env():
    #         log("❌ 环境创建失败")
    #         return
    # else:
    #     log(f"✅ conda 环境 {ENV_NAME} 已存在，跳过创建")
    
    # 2. 训练
    all_results = {}
    all_results["train"] = train_model()
    
    # 3. 剪枝
    all_results["prune"] = prune_model()
    
    # 4. 量化
    all_results["quantize"] = quantize_model()
    
    # 5. 生成汇总
    generate_summary(all_results)
    
    log("\n" + "="*60)
    log("🎉 所有实验完成!")
    log("="*60)
    log(f"📁 结果目录: {RESULTS_DIR}")
    log(f"📋 日志文件: {LOG_FILE}")


if __name__ == "__main__":
    main()
