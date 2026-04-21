# Step 2: 剪枝MNIST SpatialNet模型
# 运行: python step2_prune.py

import torch
import copy

# 导入模型
from step1_spatialnet_mnist import SpatialNet4Layer

def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in dataloader:
            _, predicted = model(data).max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

def main():
    print("=" * 60)
    print("Step 2: 剪枝MNIST SpatialNet模型")
    print("=" * 60)
    
    # 加载MNIST数据
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    val_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False
    )
    
    # 加载训练好的模型
    print("\n加载训练好的模型...")
    checkpoint = torch.load('mnist_spatialnet_trained.pth', weights_only=False)
    
    model = SpatialNet4Layer(in_channels=1, num_classes=10)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型，参数量: {checkpoint['param_count']:,}")
    print(f"原始准确率: {checkpoint['accuracy']:.2f}%")
    
    # 评估基线
    baseline_acc = evaluate(model, val_loader)
    print(f"\n基线准确率(重新评估): {baseline_acc:.2f}%")
    
    # Level剪枝
    print("\n" + "=" * 60)
    print("Level Pruner (稀疏度: 0.5)")
    print("=" * 60)
    
    from nni.compression.pruning import LevelPruner
    
    pruned_model = copy.deepcopy(model)
    config_list = [
        {'sparsity': 0.5, 'op_types': ['Conv2d', 'Linear']}
    ]
    
    print("创建LevelPruner...")
    pruner = LevelPruner(pruned_model, config_list)
    
    print("执行剪枝...")
    pruned_model, masks = pruner.compress()
    
    pruned_acc = evaluate(pruned_model, val_loader)
    print(f"\n剪枝后准确率: {pruned_acc:.2f}%")
    print(f"准确率变化: {baseline_acc:.2f}% -> {pruned_acc:.2f}% ({pruned_acc - baseline_acc:+.2f}%)")
    
    # 保存剪枝后的模型
    torch.save({
        'model_state_dict': pruned_model.state_dict(),
        'accuracy': pruned_acc,
        'masks': masks
    }, 'mnist_spatialnet_pruned.pth')
    
    print(f"\n剪枝后模型已保存到: mnist_spatialnet_pruned.pth")
    print("=" * 60)
    print("Step 2 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()