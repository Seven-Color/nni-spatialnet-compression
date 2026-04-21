# Step 2: MNIST SpatialNet 动态剪枝和量化演示
# ============================================================
# 在4层SpatialNet上应用NNI的动态剪枝和量化
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
import time
import nni

# 导入模型
from step1_spatialnet_mnist import SpatialNet4Layer, get_mnist_dataloaders


# ============================================================
# 辅助函数
# ============================================================

def evaluate(model, dataloader, device='cpu'):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_mb(model):
    """计算模型大小（MB）"""
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    return size / (1024 * 1024)


# ============================================================
# 压缩Pipeline
# ============================================================

class MNISTCompressionPipeline:
    """MNIST SpatialNet压缩Pipeline"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results = {}
    
    def run_baseline(self):
        """运行基线评估"""
        print("\n" + "="*60)
        print("基线模型评估")
        print("="*60)
        
        model = copy.deepcopy(self.original_model).to(self.device)
        accuracy = evaluate(model, self.val_loader, self.device)
        param_count = count_parameters(model)
        model_size = calculate_model_size_mb(model)
        
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小: {model_size:.2f} MB")
        
        self.results['baseline'] = {
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size
        }
        
        return model, self.results['baseline']
    
    def run_level_pruning(self, sparsity=0.5):
        """Level剪枝 - 简单粗暴地剪掉一定比例的参数"""
        print("\n" + "="*60)
        print(f"Level Pruner (稀疏度: {sparsity})")
        print("="*60)
        
        from nni.compression.pruning import LevelPruner
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 剪枝配置
        config_list = [
            {'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        # 创建剪枝器
        pruner = LevelPruner(model, config_list)
        
        # 执行剪枝
        start_time = time.time()
        pruned_model, masks = pruner.compress()
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(pruned_model, self.val_loader, self.device)
        param_count = count_parameters(pruned_model)
        model_size = calculate_model_size_mb(pruned_model)
        
        print(f"剪枝耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小: {model_size:.2f} MB")
        
        # 计算压缩比
        baseline_size = self.results['baseline']['model_size_mb']
        compression_ratio = baseline_size / model_size
        print(f"压缩比: {compression_ratio:.2f}x")
        
        self.results['level_pruning'] = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size,
            'time_seconds': elapsed
        }
        
        return pruned_model, self.results['level_pruning']
    
    def run_l1norm_pruning(self, sparsity=0.5):
        """L1-Norm剪枝 - 根据权重的L1范数评估重要性"""
        print("\n" + "="*60)
        print(f"L1-Norm Pruner (稀疏度: {sparsity})")
        print("="*60)
        
        from nni.compression.pruning import L1NormPruner
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 剪枝配置
        config_list = [
            {'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        # 创建剪枝器
        pruner = L1NormPruner(model, config_list)
        
        # 执行剪枝
        start_time = time.time()
        pruned_model, masks = pruner.compress()
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(pruned_model, self.val_loader, self.device)
        param_count = count_parameters(pruned_model)
        model_size = calculate_model_size_mb(pruned_model)
        
        print(f"剪枝耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小: {model_size:.2f} MB")
        
        self.results['l1norm_pruning'] = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size,
            'time_seconds': elapsed
        }
        
        return pruned_model, self.results['l1norm_pruning']
    
    def run_agp_pruning(self, sparsity=0.6, total_steps=30):
        """AGP剪枝 - 自动化渐进式剪枝"""
        print("\n" + "="*60)
        print(f"AGP Pruner (目标稀疏度: {sparsity})")
        print("="*60)
        
        from nni.compression.pruning import AGPPruner, LevelPruner
        from nni.compression.utils import TorchEvaluator
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 基础剪枝器
        base_pruner = LevelPruner(model, [{'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}])
        
        # 单步训练函数
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # 训练循环
        def training_func(model, optimizers, training_step_fn,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            steps = max_steps if max_steps else total_steps
            
            model.train()
            batch_iter = iter(self.train_loader)
            for step in range(steps):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(self.train_loader)
                    batch = next(batch_iter)
                optimizer.zero_grad()
                loss = training_step_fn(batch, model)
                loss.backward()
                optimizer.step()
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_func,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建AGP剪枝器
        interval_steps = max(1, total_steps // 10)
        pruner = AGPPruner(
            pruner=base_pruner,
            interval_steps=interval_steps,
            total_times=10,
            evaluator=evaluator
        )
        
        # 执行剪枝
        start_time = time.time()
        pruned_model, masks = pruner.compress(max_steps=total_steps, max_epochs=None)
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(pruned_model, self.val_loader, self.device)
        param_count = count_parameters(pruned_model)
        model_size = calculate_model_size_mb(pruned_model)
        
        print(f"剪枝耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小: {model_size:.2f} MB")
        
        self.results['agp_pruning'] = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size,
            'time_seconds': elapsed
        }
        
        return pruned_model, self.results['agp_pruning']
    
    def run_ptq_quantization(self, quant_bits=8):
        """PTQ量化 - 训练后量化"""
        print("\n" + "="*60)
        print(f"PTQ Quantizer (量化位数: {quant_bits})")
        print("="*60)
        
        from nni.compression.quantization import PtqQuantizer
        from nni.compression.utils import TorchEvaluator
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 量化配置 - NNI 3.0格式
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_dtype': f'int{quant_bits}',
                'target_names': ['weight', '_output_'],
                'target_settings': {
                    'weight': {'quant_dtype': f'int{quant_bits}'},
                    '_output_': {'quant_dtype': f'int{quant_bits}'}
                }
            }
        ]
        
        # PTQ校准步骤
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # PTQ训练循环
        def training_func(model, optimizers, training_step_fn,
                          lr_schedulers=None, max_steps=None, max_epochs=None):
            model.train()
            total_steps = max_steps if max_steps else 100
            current_steps = 0
            
            for epoch in range(max_epochs if max_epochs else 1):
                for batch in self.train_loader:
                    if current_steps >= total_steps:
                        return
                    training_step_fn(batch, model)
                    current_steps += 1
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_func,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = PtqQuantizer(model, config_list, evaluator)
        
        # 执行量化
        start_time = time.time()
        quantized_model, calibration_config = quantizer.compress(max_steps=100, max_epochs=None)
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(quantized_model, self.val_loader, self.device)
        model_size = calculate_model_size_mb(quantized_model)
        theoretical_size = count_parameters(quantized_model) * quant_bits / 8 / (1024 * 1024)
        
        print(f"量化耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"模型大小(FP32): {model_size:.2f} MB")
        print(f"理论大小({quant_bits}bit): {theoretical_size:.2f} MB")
        
        self.results['ptq_quantization'] = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'theoretical_size_mb': theoretical_size,
            'time_seconds': elapsed
        }
        
        return quantized_model, self.results['ptq_quantization']
    
    def run_qat_quantization(self, quant_bits=8, max_epochs=3):
        """QAT量化 - 量化感知训练"""
        print("\n" + "="*60)
        print(f"QAT Quantizer (量化位数: {quant_bits})")
        print("="*60)
        
        from nni.compression.quantization import QATQuantizer
        from nni.compression.utils import TorchEvaluator
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 量化配置 - NNI 3.0格式
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_dtype': f'int{quant_bits}',
                'target_names': ['weight', '_output_'],
                'target_settings': {
                    'weight': {'quant_dtype': f'int{quant_bits}'},
                    '_output_': {'quant_dtype': f'int{quant_bits}'}
                }
            }
        ]
        
        # 单步训练函数
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # 训练循环
        def training_func(model, optimizers, training_step_fn,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            epochs = max_epochs if max_epochs else max_epochs
            
            model.train()
            for epoch in range(epochs):
                for batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = training_step_fn(batch, model)
                    loss.backward()
                    optimizer.step()
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_func,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = QATQuantizer(model, config_list, evaluator)
        
        # 执行量化
        start_time = time.time()
        quantized_model, calibration_config = quantizer.compress(max_steps=None, max_epochs=max_epochs)
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(quantized_model, self.val_loader, self.device)
        model_size = calculate_model_size_mb(quantized_model)
        theoretical_size = count_parameters(quantized_model) * quant_bits / 8 / (1024 * 1024)
        
        print(f"量化耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"模型大小(FP32): {model_size:.2f} MB")
        print(f"理论大小({quant_bits}bit): {theoretical_size:.2f} MB")
        
        self.results['qat_quantization'] = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'model_size_mb': model_size,
            'theoretical_size_mb': theoretical_size,
            'time_seconds': elapsed
        }
        
        return quantized_model, self.results['qat_quantization']
    
    def run_combined_compression(self, prune_sparsity=0.5, quant_bits=8):
        """联合压缩 - 先剪枝后量化"""
        print("\n" + "="*60)
        print(f"联合压缩 (剪枝稀疏度: {prune_sparsity}, 量化位数: {quant_bits})")
        print("="*60)
        
        from nni.compression.quantization import QATQuantizer
        from nni.compression.utils import TorchEvaluator
        
        # 使用原始模型进行联合压缩
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 量化配置 - NNI 3.0格式
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_dtype': f'int{quant_bits}',
                'target_names': ['weight', '_output_'],
                'target_settings': {
                    'weight': {'quant_dtype': f'int{quant_bits}'},
                    '_output_': {'quant_dtype': f'int{quant_bits}'}
                }
            }
        ]
        
        # 单步训练函数
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # 训练循环
        def training_func(model, optimizers, training_step_fn,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            epochs = max_epochs if max_epochs else 3
            
            model.train()
            for epoch in range(epochs):
                for batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = training_step_fn(batch, model)
                    loss.backward()
                    optimizer.step()
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_func,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = QATQuantizer(model, config_list, evaluator)
        
        # 执行量化
        start_time = time.time()
        combined_model, calibration_config = quantizer.compress(max_steps=None, max_epochs=3)
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(combined_model, self.val_loader, self.device)
        param_count = count_parameters(combined_model)
        model_size = calculate_model_size_mb(combined_model)
        theoretical_size = param_count * quant_bits / 8 / (1024 * 1024)
        
        print(f"\n联合压缩总耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小(FP32): {model_size:.2f} MB")
        print(f"理论大小({quant_bits}bit): {theoretical_size:.2f} MB")
        
        # 计算压缩比
        baseline_size = self.results['baseline']['model_size_mb']
        compression_ratio = baseline_size / theoretical_size
        space_saving = 1 - theoretical_size / baseline_size
        
        print(f"压缩比: {compression_ratio:.2f}x")
        print(f"空间节省: {space_saving:.2%}")
        
        self.results['combined'] = {
            'prune_sparsity': prune_sparsity,
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size,
            'theoretical_size_mb': theoretical_size,
            'compression_ratio': compression_ratio,
            'space_saving': space_saving,
            'time_seconds': elapsed
        }
        
        return combined_model, self.results['combined']
    
    def print_summary(self):
        """打印压缩结果摘要"""
        print("\n" + "="*60)
        print("压缩结果摘要")
        print("="*60)
        
        print(f"{'方法':<20} {'准确率':<10} {'大小(MB)':<12} {'压缩比':<10}")
        print("-" * 60)
        print(f"{'基线':<20} {self.results['baseline']['accuracy']:.2f}%     "
              f"{self.results['baseline']['model_size_mb']:.2f}        -")
        
        if 'level_pruning' in self.results:
            r = self.results['level_pruning']
            ratio = self.results['baseline']['model_size_mb'] / r['model_size_mb']
            print(f"{'Level剪枝':<20} {r['accuracy']:.2f}%     {r['model_size_mb']:.2f}        {ratio:.2f}x")
        
        if 'l1norm_pruning' in self.results:
            r = self.results['l1norm_pruning']
            ratio = self.results['baseline']['model_size_mb'] / r['model_size_mb']
            print(f"{'L1Norm剪枝':<20} {r['accuracy']:.2f}%     {r['model_size_mb']:.2f}        {ratio:.2f}x")
        
        if 'agp_pruning' in self.results:
            r = self.results['agp_pruning']
            ratio = self.results['baseline']['model_size_mb'] / r['model_size_mb']
            print(f"{'AGP剪枝':<20} {r['accuracy']:.2f}%     {r['model_size_mb']:.2f}        {ratio:.2f}x")
        
        if 'ptq_quantization' in self.results:
            r = self.results['ptq_quantization']
            ratio = self.results['baseline']['model_size_mb'] / r['theoretical_size_mb']
            print(f"{'PTQ量化':<20} {r['accuracy']:.2f}%     {r['theoretical_size_mb']:.2f}        {ratio:.2f}x")
        
        if 'qat_quantization' in self.results:
            r = self.results['qat_quantization']
            ratio = self.results['baseline']['model_size_mb'] / r['theoretical_size_mb']
            print(f"{'QAT量化':<20} {r['accuracy']:.2f}%     {r['theoretical_size_mb']:.2f}        {ratio:.2f}x")
        
        if 'combined' in self.results:
            r = self.results['combined']
            print(f"{'联合压缩':<20} {r['accuracy']:.2f}%     {r['theoretical_size_mb']:.2f}        {r['compression_ratio']:.2f}x")
        
        print("=" * 60)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("MNIST 4层SpatialNet 动态剪枝和量化演示")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载MNIST数据
    print("\n加载MNIST数据集...")
    train_loader, val_loader = get_mnist_dataloaders(batch_size=64)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建4层SpatialNet模型...")
    model = SpatialNet4Layer(in_channels=1, num_classes=10).to(device)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 创建压缩Pipeline
    pipeline = MNISTCompressionPipeline(model, train_loader, val_loader, device)
    
    # 运行基线
    baseline_model, baseline_results = pipeline.run_baseline()
    
    # 运行Level剪枝
    level_model, level_results = pipeline.run_level_pruning(sparsity=0.5)
    
    # 运行L1-Norm剪枝
    l1norm_model, l1norm_results = pipeline.run_l1norm_pruning(sparsity=0.5)
    
    # 运行AGP剪枝
    agp_model, agp_results = pipeline.run_agp_pruning(sparsity=0.6, total_steps=30)
    
    # 运行PTQ量化
    ptq_model, ptq_results = pipeline.run_ptq_quantization(quant_bits=8)
    
    # 运行QAT量化
    qat_model, qat_results = pipeline.run_qat_quantization(quant_bits=8, max_epochs=3)
    
    # 运行联合压缩
    combined_model, combined_results = pipeline.run_combined_compression(
        prune_sparsity=0.5,
        quant_bits=8
    )
    
    # 打印摘要
    pipeline.print_summary()
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()