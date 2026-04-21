# Step 3: 量化MNIST SpatialNet模型
# 运行: python step3_quantize.py

import torch
import torch.nn.functional as F
import nni

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
    print("Step 3: 量化MNIST SpatialNet模型")
    print("=" * 60)
    
    # 加载MNIST数据
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = torchvision.datasets.MNIST(
        root='./data/mnist',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
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
    
    # 评估基线
    baseline_acc = evaluate(model, val_loader)
    print(f"基线准确率: {baseline_acc:.2f}%")
    
    # QAT量化
    print("\n" + "=" * 60)
    print("QAT Quantizer (8-bit, 1 epoch)")
    print("=" * 60)
    
    from nni.compression.quantization import QATQuantizer
    from nni.compression.utils import TorchEvaluator
    
    # 复制模型
    quant_model = model
    
    # 量化配置 - NNI 3.0格式
    config_list = [
        {
            'op_types': ['Conv2d', 'Linear'],
            'quant_dtype': 'int8',
            'target_names': ['weight', '_output_'],
            'target_settings': {
                'weight': {'quant_dtype': 'int8'},
                '_output_': {'quant_dtype': 'int8'}
            }
        }
    ]
    
    # 单步训练函数
    def training_step(batch, model):
        images, labels = batch
        outputs = model(images)
        return F.cross_entropy(outputs, labels)
    
    # 训练循环
    def training_func(model, optimizers, training_step_fn,
                     lr_schedulers=None, max_steps=None, max_epochs=None):
        optimizer = optimizers
        model.train()
        for epoch in range(max_epochs if max_epochs else 1):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = training_step_fn(batch, model)
                loss.backward()
                optimizer.step()
    
    # 评估函数
    def evaluating_func(m):
        return evaluate(m, val_loader)
    
    # 创建评估器
    traced_optimizer = nni.trace(torch.optim.Adam)(quant_model.parameters(), lr=0.001)
    evaluator = TorchEvaluator(
        training_func=training_func,
        optimizers=traced_optimizer,
        training_step=training_step,
        evaluating_func=evaluating_func
    )
    
    # 创建量化器
    print("创建QATQuantizer...")
    quantizer = QATQuantizer(quant_model, config_list, evaluator)
    
    # 执行量化
    print("执行QAT量化 (1 epoch)...")
    quantized_model, calibration_config = quantizer.compress(max_steps=None, max_epochs=1)
    
    # 评估量化后的模型
    qat_acc = evaluate(quantized_model, val_loader)
    
    print(f"\n量化后准确率: {qat_acc:.2f}%")
    print(f"准确率变化: {baseline_acc:.2f}% -> {qat_acc:.2f}% ({qat_acc - baseline_acc:+.2f}%)")
    
    # 计算理论压缩比
    param_count = checkpoint['param_count']
    fp32_size = param_count * 32 / 8 / (1024 * 1024)  # MB
    int8_size = param_count * 8 / 8 / (1024 * 1024)  # MB
    print(f"\n模型大小: {fp32_size:.2f} MB (FP32) -> {int8_size:.2f} MB (INT8)")
    print(f"压缩比: {fp32_size / int8_size:.2f}x")
    
    # 保存量化后的模型
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'accuracy': qat_acc,
        'calibration_config': calibration_config
    }, 'mnist_spatialnet_quantized.pth')
    
    print(f"\n量化后模型已保存到: mnist_spatialnet_quantized.pth")
    print("=" * 60)
    print("Step 3 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()