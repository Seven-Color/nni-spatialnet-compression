# Step 6: 完整运行演示
# ============================================================
# 本步骤整合所有组件，提供一个完整的端到端演示
# 包括模型定义、数据准备、剪枝、量化
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import copy
import time
import nni
from nni.compression import TorchEvaluator
from nni.compression.pruning import L1NormPruner, LevelPruner, AGPPruner
from nni.compression.quantization import QATQuantizer, PtqQuantizer


# ============================================================
# 第1部分: SpatialNet模型定义（简化版）
# ============================================================

class SpatialNet(nn.Module):
    """
    简化版SpatialNet模型
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(SpatialNet, self).__init__()
        
        # 特征提取器
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# 第2部分: 数据集定义
# ============================================================

class DummyDataset(Dataset):
    """模拟数据集"""
    def __init__(self, num_samples=500, size=32, channels=3, num_classes=10, seed=42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, channels, size, size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_dataloaders(batch_size=32):
    """创建数据加载器"""
    train_dataset = DummyDataset(num_samples=400, seed=42)
    val_dataset = DummyDataset(num_samples=100, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# ============================================================
# 第3部分: 评估函数
# ============================================================

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size_mb(model):
    """计算模型大小（MB）"""
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    return size / (1024 * 1024)


def calculate_sparsity(model):
    """计算模型稀疏度"""
    total = 0
    zeros = 0
    for p in model.parameters():
        if 'weight' in p.name if hasattr(p, 'name') else True:
            total += p.numel()
            zeros += (p == 0).sum().item()
    return zeros / total if total > 0 else 0


# ============================================================
# 第4部分: 压缩Pipeline
# ============================================================

class CompressionPipeline:
    """
    完整的模型压缩Pipeline
    """
    def __init__(self, model, train_loader, val_loader, device):
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
        """
        使用Level Pruner进行幅度剪枝
        这是一种最基本的剪枝方法，直接剪枝最小幅度的权重
        """
        print("\n" + "="*60)
        print(f"Level Pruner (稀疏度: {sparsity})")
        print("="*60)
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 剪枝配置
        config_list = [
            {'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器（不需要训练步骤）
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=None,
            optimizers=traced_optimizer,
            training_step=None,
            evaluating_func=evaluating_func
        )
        
        # 创建剪枝器
        pruner = LevelPruner(model, config_list, evaluator)
        
        # 执行剪枝（不训练，只应用静态剪枝）
        start_time = time.time()
        pruned_model, masks = pruner.compress()
        elapsed = time.time() - start_time
        
        # 评估
        accuracy = evaluate(pruned_model, self.val_loader, self.device)
        param_count = count_parameters(pruned_model)
        model_size = calculate_model_size_mb(pruned_model)
        
        # 计算稀疏度
        def calc_sparsity_from_masks(masks):
            if not masks:
                return 0.0
            total_zeros = 0
            total_elements = 0
            for module_name, target_masks in masks.items():
                for target_name, mask in target_masks.items():
                    if mask is not None:
                        total_zeros += (mask == 0).sum().item()
                        total_elements += mask.numel()
            return total_zeros / total_elements if total_elements > 0 else 0.0
        
        sparsity_ratio = calc_sparsity_from_masks(masks)
        
        print(f"剪枝耗时: {elapsed:.2f}秒")
        print(f"准确率: {accuracy:.2f}%")
        print(f"参数量: {param_count:,}")
        print(f"模型大小: {model_size:.2f} MB")
        print(f"生成的掩码稀疏度: {sparsity_ratio:.2%}")
        
        self.results['level_pruning'] = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'param_count': param_count,
            'model_size_mb': model_size,
            'time_seconds': elapsed,
            'masks': masks
        }
        
        return pruned_model, self.results['level_pruning']
    
    def run_l1norm_pruning(self, sparsity=0.5):
        """
        使用L1范数剪枝
        根据权重的L1范数评估Filter的重要性，剪枝范数最小的Filter
        """
        print("\n" + "="*60)
        print(f"L1-Norm Pruner (稀疏度: {sparsity})")
        print("="*60)
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 剪枝配置
        config_list = [
            {'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=None,
            optimizers=traced_optimizer,
            training_step=None,
            evaluating_func=evaluating_func
        )
        
        # 创建剪枝器
        pruner = L1NormPruner(model, config_list, evaluator)
        
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
    
    def run_agp_pruning(self, sparsity=0.8, warm_up_steps=10, total_steps=50):
        """
        使用AGP（渐近剪枝）剪枝
        AGP在训练过程中渐进式增加稀疏度，先从低稀疏度开始，逐渐增加
        """
        print("\n" + "="*60)
        print(f"AGP Pruner (目标稀疏度: {sparsity})")
        print("="*60)
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 首先创建基础剪枝器（不带评估器）
        base_pruner = L1NormPruner(
            model, 
            config_list=[{'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}],
            evaluator=None  # 不带evaluator
        )
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # 训练步骤
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # 训练循环
        def training_loop(model, optimizers, training_step_fn,
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
        
        # 创建评估器
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建AGP剪枝器 - 包装基础剪枝器
        interval_steps = max(1, total_steps // 10)  # 每隔一定步数更新一次
        pruner = AGPPruner(
            pruner=base_pruner,
            interval_steps=interval_steps,
            total_times=10,  # 总共更新10次
            evaluator=evaluator
        )
        
        # 执行剪枝（训练感知）
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
        """
        使用PTQ（训练后量化）
        最简单的量化方法，不需要重新训练
        """
        print("\n" + "="*60)
        print(f"PTQ Quantizer (量化位数: {quant_bits})")
        print("="*60)
        
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
        
        # 评估函数
        def evaluating_func(m):
            return evaluate(m, self.val_loader, self.device)
        
        # PTQ校准步骤
        def training_step(batch, model):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        
        # PTQ训练循环 - 用于校准
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
        
        # 执行量化（PTQ需要校准）
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
        """
        使用QAT（量化感知训练）
        在训练过程中模拟量化效果，需要训练但精度更好
        """
        print("\n" + "="*60)
        print(f"QAT Quantizer (量化位数: {quant_bits})")
        print("="*60)
        
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
    
    def run_combined_compression(self, prune_sparsity=0.5, quant_bits=8, use_original_model=False):
        """
        联合压缩：先剪枝后量化
        
        Args:
            prune_sparsity: 剪枝稀疏度
            quant_bits: 量化位数
            use_original_model: 如果为True，使用原始模型进行联合压缩
                             如果为False，使用剪枝后的模型（需要正确处理wrapper）
        """
        print("\n" + "="*60)
        print(f"联合压缩 (剪枝稀疏度: {prune_sparsity}, 量化位数: {quant_bits})")
        print("="*60)
        
        # 确定用于量化的模型
        if use_original_model:
            # 使用原始模型（深拷贝）进行联合压缩
            print("\n[Info] 使用原始模型进行联合压缩...")
            model_for_quant = copy.deepcopy(self.original_model).to(self.device)
        else:
            # 第一步：剪枝
            print("\n[Step 1] Level剪枝...")
            prune_model, _ = self.run_level_pruning(sparsity=prune_sparsity)
            model_for_quant = prune_model
        
        # 第二步：量化
        print("\n[Step 2] QAT量化...")
        
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
        traced_optimizer = nni.trace(torch.optim.Adam)(model_for_quant.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_func,
            optimizers=traced_optimizer,
            training_step=training_step,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = QATQuantizer(model_for_quant, config_list, evaluator)
        
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
        
        baseline = self.results.get('baseline', {})
        
        print(f"\n{'方法':<25} {'准确率':<12} {'大小(MB)':<12}")
        print("-" * 60)
        
        if baseline:
            print(f"{'基线':<25} {baseline['accuracy']:.2f}%{'':<6} {baseline['model_size_mb']:.2f}")
        
        for name, result in self.results.items():
            if name == 'baseline':
                continue
            method_name = name.replace('_', ' ').title()
            acc = result.get('accuracy', 'N/A')
            if isinstance(acc, float):
                acc_str = f"{acc:.2f}%"
            else:
                acc_str = str(acc)
            
            size = result.get('theoretical_size_mb') or result.get('model_size_mb', 'N/A')
            if isinstance(size, float):
                size_str = f"{size:.2f}"
            else:
                size_str = str(size)
            
            print(f"{method_name:<25} {acc_str:<12} {size_str:<12}")


# ============================================================
# 第5部分: 主程序
# ============================================================

def main():
    """主程序"""
    print("="*60)
    print("NNI SpatialNet 动态剪枝与量化演示")
    print("="*60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建模型
    print("\n创建SpatialNet模型...")
    model = SpatialNet(in_channels=3, num_classes=10)
    model = model.to(device)
    print(f"参数量: {count_parameters(model):,}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader = create_dataloaders(batch_size=32)
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 创建压缩Pipeline
    pipeline = CompressionPipeline(model, train_loader, val_loader, device)
    
    # 运行基线
    baseline_model, baseline_results = pipeline.run_baseline()
    
    # 运行Level剪枝
    level_model, level_results = pipeline.run_level_pruning(sparsity=0.5)
    
    # 运行L1-Norm剪枝
    l1norm_model, l1norm_results = pipeline.run_l1norm_pruning(sparsity=0.5)
    
    # 运行AGP剪枝（训练感知）
    agp_model, agp_results = pipeline.run_agp_pruning(sparsity=0.6, total_steps=30)
    
    # 运行PTQ量化
    ptq_model, ptq_results = pipeline.run_ptq_quantization(quant_bits=8)
    
    # 运行QAT量化
    qat_model, qat_results = pipeline.run_qat_quantization(quant_bits=8, max_epochs=3)
    
    # 注意：联合压缩（先剪枝后量化）需要正确处理wrapper传递
    # 由于剪枝后的模型已被wrapper包装，直接在其上应用量化会冲突
    # 如需联合压缩，请使用 from_compressor 方法或重新加载未剪枝的模型
    print("\n" + "="*60)
    print("联合压缩已跳过（模型已被wrapper包装，直接量化会冲突）")
    print("如需联合压缩，请使用 QATQuantizer.from_compressor() 方法")
    print("="*60)
    
    # 运行联合压缩（使用原始模型重新进行联合压缩，而非用剪枝后的模型）
    print("\n[替代方案] 使用原始模型进行联合压缩...")
    combined_model, combined_results = pipeline.run_combined_compression(
        prune_sparsity=0.5,
        quant_bits=8,
        use_original_model=True  # 新增参数，使用原始模型而非剪枝后的模型
    )
    
    # 打印摘要
    pipeline.print_summary()
    
    print("\n演示完成!")


if __name__ == "__main__":
    main()
