"""量化模块"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


def load_data(config):
    """加载数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_ds = datasets.MNIST(
        config['dataset']['root'], train=False, download=True, transform=transform
    )
    return DataLoader(test_ds, batch_size=512, shuffle=False)


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct, total = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return 100. * correct / total


def quantize_dynamic(model):
    """动态量化"""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )


def quantize_static(model, calibration_loader, device):
    """静态量化 (PTQ)"""
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # 校准
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            if i >= 5:  # 使用前5个batch校准
                break
            model(data.to(device))
    
    quantized_model = torch.quantization.convert(model, inplace=False)
    return quantized_model


def quantize_qat(model, train_loader, device, epochs=1):
    """量化感知训练 (QAT)"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 插入 QAT 指定
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    quantized_model = torch.quantization.convert(model, inplace=False)
    return quantized_model


def quantize_main(config):
    """量化主函数"""
    print("\n" + "="*60)
    print("⚡ 量化阶段")
    print("="*60)
    
    device = torch.device("cpu")
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        config['dataset']['root'], train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 加载基线模型
    model = build_model(config)
    baseline_path = os.path.join(config['output']['dir'], "baseline.pt")
    model.load_state_dict(torch.load(baseline_path, weights_only=False))
    
    # 评估基线
    baseline_acc = evaluate(model, test_loader, device)
    print(f"基线准确率: {baseline_acc:.2f}%")
    
    # 执行量化
    quant_cfg = config['quantize']
    method = quant_cfg.get('method', 'dynamic_int8')
    print(f"量化方法: {method}")
    
    if method == 'dynamic_int8':
        quantized_model = quantize_dynamic(model)
    elif method == 'static_int8':
        calibration_loader = DataLoader(train_ds, batch_size=256, shuffle=False)
        quantized_model = quantize_static(model, calibration_loader, device)
    elif method == 'qat':
        quantized_model = quantize_qat(model, train_loader, device, epochs=1)
    else:
        print(f"未知的量化方法: {method}")
        return {}
    
    # 评估量化后
    quant_acc = evaluate(quantized_model, test_loader, device)
    print(f"量化后准确率: {quant_acc:.2f}%")
    print(f"准确率变化: {quant_acc - baseline_acc:+.2f}%")
    
    # 保存
    output_dir = config['output']['dir']
    quant_path = os.path.join(output_dir, "quantized.pt")
    torch.save(quantized_model.state_dict(), quant_path)
    
    result = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(quant_acc, 2),
        "accuracy_drop": round(baseline_acc - quant_acc, 2),
        "method": method,
        "model_path": quant_path
    }
    
    result_path = os.path.join(output_dir, "quantize_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 量化完成!")
    print(f"   模型保存: {quant_path}")
    
    return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    quantize_main(config)
