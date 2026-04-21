"""剪枝模块"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import prune

from model import build_model, count_params


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


def get_pruner(method):
    """获取剪枝方法"""
    pruners = {
        'l1_unstructured': lambda m, n, a: prune.l1_unstructured(m, n, amount=a),
        'random_unstructured': lambda m, n, a: prune.random_unstructured(m, n, amount=a),
        'ln_structured': lambda m, n, a, d: prune.ln_structured(m, n, amount=a, n=2, dim=d),
    }
    return pruners.get(method)


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


def prune_model(model, config):
    """执行剪枝"""
    prune_cfg = config['prune']
    
    if not prune_cfg.get('enable', True):
        print("剪枝已禁用")
        return model, {}
    
    sparsity = prune_cfg.get('sparsity', 0.5)
    method = prune_cfg.get('method', 'l1_unstructured')
    
    print(f"\n执行剪枝: {method}, 稀疏度={sparsity}")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            pruner = get_pruner(method)
            if pruner:
                if 'ln_structured' in method:
                    # structured pruning 需要 dim 参数
                    pruner(module, 'weight', sparsity, dim=0)
                else:
                    pruner(module, 'weight', sparsity)
    
    return model, {"sparsity": sparsity, "method": method}


def prune_main(config):
    """剪枝主函数"""
    print("\n" + "="*60)
    print("✂️ 剪枝阶段")
    print("="*60)
    
    device = torch.device("cpu")
    test_loader = load_data(config)
    
    # 加载基线模型
    model = build_model(config)
    baseline_path = os.path.join(config['output']['dir'], "baseline.pt")
    model.load_state_dict(torch.load(baseline_path, weights_only=False))
    
    # 评估基线
    baseline_acc = evaluate(model, test_loader, device)
    params = count_params(model)
    print(f"基线准确率: {baseline_acc:.2f}%")
    print(f"参数量: {params:,}")
    
    # 执行剪枝
    pruned_model, prune_info = prune_model(model, config)
    
    # 评估剪枝后
    pruned_acc = evaluate(pruned_model, test_loader, device)
    print(f"剪枝后准确率: {pruned_acc:.2f}%")
    print(f"准确率变化: {pruned_acc - baseline_acc:+.2f}%")
    
    # 保存
    output_dir = config['output']['dir']
    pruned_path = os.path.join(output_dir, "pruned.pt")
    torch.save(pruned_model.state_dict(), pruned_path)
    
    result = {
        "baseline_accuracy": round(baseline_acc, 2),
        "accuracy": round(pruned_acc, 2),
        "params": params,
        "accuracy_drop": round(baseline_acc - pruned_acc, 2),
        **prune_info,
        "model_path": pruned_path
    }
    
    result_path = os.path.join(output_dir, "prune_results.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ 剪枝完成!")
    print(f"   模型保存: {pruned_path}")
    
    return result


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    prune_main(config)
