# Step 5: 联合剪枝与量化
# ============================================================
# 本步骤演示如何结合剪枝和量化来实现更大幅度的模型压缩
# 包括Pipeline的串行和并行组合方式
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import copy
import nni
from nni.compression import TorchEvaluator
from nni.compression.pruning import (
    LevelPruner, L1NormPruner, FPGMPruner, TaylorPruner
)
from nni.compression.quantization import (
    QATQuantizer, PtqQuantizer, LsqQuantizer
)
from nni.compression.utils import auto_set_denpendency_group_ids


class CombinedCompressionExperiment:
    """
    联合压缩实验管理器
    
    支持两种Pipeline组合方式：
    1. 串行（Sequential）: 先剪枝后量化，或先量化后剪枝
    2. 融合（Fusion）: 同时应用剪枝和量化
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device = None):
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cpu')
        self.results = {}
    
    def _get_training_step(self):
        """创建训练步骤"""
        def training_step(batch, model):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        return training_step
    
    def _get_training_loop(self, max_epochs):
        """创建训练循环"""
        def training_loop(model, optimizers, training_step,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            epochs = max_epochs if max_epochs else max_epochs
            
            model.train()
            for epoch in range(epochs):
                for batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = training_step(batch, model)
                    loss.backward()
                    optimizer.step()
        return training_loop
    
    def _get_evaluating_func(self):
        """创建评估函数"""
        def evaluating_func(model):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total
        return evaluating_func
    
    def baseline_evaluation(self) -> Tuple[float, float]:
        """评估原始模型"""
        model = self.original_model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def prune_then_quantize(self, 
                            prune_sparsity: float = 0.5,
                            quant_bits: int = 8,
                            prune_max_steps: int = 100,
                            quant_max_epochs: int = 3) -> Tuple[nn.Module, Dict]:
        """
        Pipeline 1: 先剪枝后量化
        
        步骤：
        1. 使用剪枝器压缩模型
        2. 使用量化器对剪枝后的模型进行量化
        
        Args:
            prune_sparsity: 剪枝稀疏度
            quant_bits: 量化位数
            prune_max_steps: 剪枝训练步数
            quant_max_epochs: 量化训练轮数
            
        Returns:
            (压缩后模型, 结果字典)
        """
        print(f"\n{'='*60}")
        print(f"Pipeline: 先剪枝后量化")
        print(f"剪枝稀疏度: {prune_sparsity}, 量化位数: {quant_bits}")
        print(f"{'='*60}")
        
        # 第一步：剪枝
        print("\n[Step 1] 执行剪枝...")
        model = copy.deepcopy(self.original_model).to(self.device)
        
        prune_config = [
            {'sparsity': prune_sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        training_step = self._get_training_step()
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 使用L1范数剪枝器
        pruner = L1NormPruner(model, prune_config, evaluator)
        pruned_model, masks = pruner.compress(max_steps=prune_max_steps)
        
        _, acc_after_prune = self._evaluate_model(pruned_model)
        print(f"剪枝后准确率: {acc_after_prune:.2f}%")
        
        # 第二步：量化
        print("\n[Step 2] 执行量化...")
        quant_config = [
            {'op_types': ['Conv2d', 'Linear'], 'quant_types': ['weight', 'output'], 
             'quant_bits': quant_bits}
        ]
        
        training_loop = self._get_training_loop(quant_max_epochs)
        traced_optimizer2 = nni.trace(torch.optim.Adam)(pruned_model.parameters(), lr=0.001)
        evaluator2 = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer2,
            evaluating_func=evaluating_func
        )
        
        quantizer = QATQuantizer(pruned_model, quant_config, evaluator2)
        quantized_model, calibration_config = quantizer.compress(max_epochs=quant_max_epochs)
        
        _, acc_final = self._evaluate_model(quantized_model)
        print(f"量化后准确率: {acc_final:.2f}%")
        
        result = {
            'method': 'prune_then_quantize',
            'prune_sparsity': prune_sparsity,
            'quant_bits': quant_bits,
            'acc_after_prune': acc_after_prune,
            'acc_final': acc_final,
            'masks': masks,
            'calibration_config': calibration_config
        }
        self.results['prune_then_quantize'] = result
        
        return quantized_model, result
    
    def quantize_then_prune(self,
                            prune_sparsity: float = 0.5,
                            quant_bits: int = 8,
                            quant_max_epochs: int = 3,
                            prune_max_steps: int = 100) -> Tuple[nn.Module, Dict]:
        """
        Pipeline 2: 先量化后剪枝
        
        注意：某些情况下量化后再剪枝可能效果更好，
        因为量化已经降低了权重的精度，剪枝的影响可能会减小
        """
        print(f"\n{'='*60}")
        print(f"Pipeline: 先量化后剪枝")
        print(f"剪枝稀疏度: {prune_sparsity}, 量化位数: {quant_bits}")
        print(f"{'='*60}")
        
        # 第一步：量化
        print("\n[Step 1] 执行量化...")
        model = copy.deepcopy(self.original_model).to(self.device)
        
        quant_config = [
            {'op_types': ['Conv2d', 'Linear'], 'quant_types': ['weight', 'output'],
             'quant_bits': quant_bits}
        ]
        
        evaluating_func = self._get_evaluating_func()
        training_loop = self._get_training_loop(quant_max_epochs)
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        quantizer = QATQuantizer(model, quant_config, evaluator)
        quantized_model, _ = quantizer.compress(max_epochs=quant_max_epochs)
        
        _, acc_after_quant = self._evaluate_model(quantized_model)
        print(f"量化后准确率: {acc_after_quant:.2f}%")
        
        # 第二步：剪枝
        print("\n[Step 2] 执行剪枝...")
        prune_config = [
            {'sparsity': prune_sparsity, 'op_types': ['Conv2d', 'Linear']}
        ]
        
        training_step = self._get_training_step()
        traced_optimizer2 = nni.trace(torch.optim.Adam)(quantized_model.parameters(), lr=0.001)
        evaluator2 = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer2,
            evaluating_func=evaluating_func
        )
        
        pruner = L1NormPruner(quantized_model, prune_config, evaluator2)
        pruned_model, masks = pruner.compress(max_steps=prune_max_steps)
        
        _, acc_final = self._evaluate_model(pruned_model)
        print(f"剪枝后准确率: {acc_final:.2f}%")
        
        result = {
            'method': 'quantize_then_prune',
            'prune_sparsity': prune_sparsity,
            'quant_bits': quant_bits,
            'acc_after_quant': acc_after_quant,
            'acc_final': acc_final,
            'masks': masks
        }
        self.results['quantize_then_prune'] = result
        
        return pruned_model, result
    
    def joint_prune_quantize(self,
                             prune_sparsity: float = 0.3,
                             quant_bits: int = 8,
                             max_epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        Pipeline 3: 联合优化（同时剪枝和量化）
        
        使用Movement Pruner等支持同时剪枝量化的压缩器
        或者通过自定义配置同时应用两种压缩
        """
        print(f"\n{'='*60}")
        print(f"Pipeline: 联合优化")
        print(f"剪枝稀疏度: {prune_sparsity}, 量化位数: {quant_bits}")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 配置同时包含剪枝和量化
        config_list = [
            {
                'sparsity': prune_sparsity,
                'quant_types': ['weight', 'output'],
                'quant_bits': quant_bits,
                'op_types': ['Conv2d', 'Linear']
            }
        ]
        
        evaluating_func = self._get_evaluating_func()
        
        def training_loop(model, optimizers, training_step,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            epochs = max_epochs if max_epochs else max_epochs
            
            model.train()
            for epoch in range(epochs):
                for batch in self.train_loader:
                    optimizer.zero_grad()
                    loss = training_step(batch, model)
                    loss.backward()
                    optimizer.step()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 使用QAT量化器（它也支持稀疏）
        quantizer = QATQuantizer(model, config_list, evaluator)
        compressed_model, calibration_config = quantizer.compress(max_epochs=max_epochs)
        
        _, accuracy = self._evaluate_model(compressed_model)
        print(f"联合压缩后准确率: {accuracy:.2f}%")
        
        result = {
            'method': 'joint_prune_quantize',
            'prune_sparsity': prune_sparsity,
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'calibration_config': calibration_config
        }
        self.results['joint_optimization'] = result
        
        return compressed_model, result
    
    def iterative_compression(self,
                              prune_sparsities: List[float] = [0.3, 0.5, 0.7],
                              quant_bits: int = 8,
                              iterations: int = 3) -> Tuple[nn.Module, Dict]:
        """
        Pipeline 4: 迭代压缩
        
        渐进式增加剪枝稀疏度，每轮剪枝后进行微调
        这种方式可以更好地平衡压缩率和精度
        
        Args:
            prune_sparsities: 各迭代轮次的剪枝稀疏度列表
            quant_bits: 量化位数
            iterations: 迭代次数
        """
        print(f"\n{'='*60}")
        print(f"Pipeline: 迭代压缩 ({iterations}次迭代)")
        print(f"剪枝稀疏度进度: {prune_sparsities}")
        print(f"量化位数: {quant_bits}")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        _, baseline_acc = self._evaluate_model(model)
        print(f"\n基线准确率: {baseline_acc:.2f}%")
        
        current_model = model
        final_result = {
            'method': 'iterative_compression',
            'iterations': iterations,
            'baseline_accuracy': baseline_acc
        }
        
        for iteration in range(iterations):
            sparsity = prune_sparsities[iteration] if iteration < len(prune_sparsities) else prune_sparsities[-1]
            print(f"\n--- 迭代 {iteration + 1}/{iterations} (稀疏度: {sparsity}) ---")
            
            # 剪枝
            print("[Step 1] 剪枝...")
            prune_config = [
                {'sparsity': sparsity, 'op_types': ['Conv2d', 'Linear']}
            ]
            
            training_step = self._get_training_step()
            evaluating_func = self._get_evaluating_func()
            
            traced_optimizer = nni.trace(torch.optim.Adam)(current_model.parameters(), lr=0.001)
            evaluator = TorchEvaluator(
                training_func=training_step,
                optimizers=traced_optimizer,
                evaluating_func=evaluating_func
            )
            
            pruner = L1NormPruner(current_model, prune_config, evaluator)
            pruned_model, _ = pruner.compress(max_steps=50)
            
            # 微调
            print("[Step 2] 微调...")
            training_loop = self._get_training_loop(max_epochs=2)
            traced_optimizer2 = nni.trace(torch.optim.Adam)(pruned_model.parameters(), lr=0.0001)
            evaluator2 = TorchEvaluator(
                training_func=training_loop,
                optimizers=traced_optimizer2,
                evaluating_func=evaluating_func
            )
            
            _, acc = self._evaluate_model(pruned_model)
            print(f"剪枝后准确率: {acc:.2f}%")
            
            current_model = pruned_model
        
        # 最终量化
        print("\n--- 最终量化 ---")
        quant_config = [
            {'op_types': ['Conv2d', 'Linear'], 'quant_types': ['weight', 'output'],
             'quant_bits': quant_bits}
        ]
        
        training_loop = self._get_training_loop(max_epochs=3)
        traced_optimizer3 = nni.trace(torch.optim.Adam)(current_model.parameters(), lr=0.001)
        evaluator3 = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer3,
            evaluating_func=evaluating_func
        )
        
        quantizer = QATQuantizer(current_model, quant_config, evaluator3)
        quantized_model, _ = quantizer.compress(max_epochs=3)
        
        _, final_acc = self._evaluate_model(quantized_model)
        print(f"最终准确率: {final_acc:.2f}%")
        
        final_result['final_accuracy'] = final_acc
        self.results['iterative_compression'] = final_result
        
        return quantized_model, final_result
    
    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """评估模型"""
        model.eval()
        model.to(self.device)
        correct = 0
        total = 0
        loss = 0.0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / total
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def compare_results(self):
        """比较所有压缩方法的结果"""
        print("\n" + "=" * 60)
        print("联合压缩方法比较")
        print("=" * 60)
        
        baseline_loss, baseline_acc = self.baseline_evaluation()
        print(f"\n基线模型: 损失={baseline_loss:.4f}, 准确率={baseline_acc:.2f}%")
        print("-" * 60)
        
        for name, result in self.results.items():
            method = result.get('method', name)
            acc = result.get('accuracy') or result.get('acc_final', 'N/A')
            sparsity = result.get('prune_sparsity', 'N/A')
            bits = result.get('quant_bits', 'N/A')
            
            if isinstance(acc, float):
                acc_str = f"{acc:.2f}%"
            else:
                acc_str = str(acc)
            
            print(f"{method}: 稀疏度={sparsity}, 位数={bits}bit, 准确率={acc_str}")


def calculate_compression_ratio(original_model: nn.Module,
                                  compressed_model: nn.Module,
                                  quant_bits: int = 8) -> Dict:
    """
    计算压缩比
    
    Args:
        original_model: 原始模型
        compressed_model: 压缩后模型
        quant_bits: 量化位数
        
    Returns:
        包含压缩统计信息的字典
    """
    # 计算原始模型大小
    original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
    
    # 计算压缩后模型大小（考虑剪枝遮罩和量化）
    compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
    
    # 估算量化后的存储大小
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    quantized_size = compressed_params * quant_bits // 8  # 字节
    
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'quantized_size_bytes': quantized_size,
        'compression_ratio': compression_ratio,
        'space_saving': 1 - (quantized_size / original_size) if original_size > 0 else 0
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("NNI联合剪枝与量化测试")
    print("=" * 60)
    
    # 导入必要的模块
    from step1_spatialnet_model import create_spatialnet_model
    from step2_data_setup import create_data_loaders
    
    # 创建设置
    print("\n创建设置和模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_loader, val_loader = create_data_loaders(
        batch_size=64,
        num_samples=500,
        image_size=32,
        in_channels=3,
        num_classes=10
    )
    
    model = create_spatialnet_model(in_channels=3, num_classes=10, width_multiplier=1.0)
    model = model.to(device)
    
    print(f"原始模型参数量: {model.get_parameter_count():,}")
    
    # 创建联合压缩实验
    experiment = CombinedCompressionExperiment(model, train_loader, val_loader, device)
    
    # 评估基线
    print("\n评估基线模型...")
    baseline_loss, baseline_acc = experiment.baseline_evaluation()
    print(f"基线损失: {baseline_loss:.4f}")
    print(f"基线准确率: {baseline_acc:.2f}%")
    
    # 测试先剪枝后量化
    print("\n" + "="*60)
    print("测试Pipeline 1: 先剪枝后量化")
    print("="*60)
    compressed_model, result = experiment.prune_then_quantize(
        prune_sparsity=0.3,
        quant_bits=8,
        prune_max_steps=50,
        quant_max_epochs=2
    )
    
    # 计算压缩比
    stats = calculate_compression_ratio(model, compressed_model, quant_bits=8)
    print(f"\n压缩统计:")
    print(f"  原始大小: {stats['original_size_bytes']:,} bytes")
    print(f"  量化后大小: {stats['quantized_size_bytes']:,} bytes")
    print(f"  压缩比: {stats['compression_ratio']:.2f}x")
    print(f"  空间节省: {stats['space_saving']:.2%}")
    
    # 比较所有结果
    experiment.compare_results()
    
    print("\n联合压缩测试完成!")
