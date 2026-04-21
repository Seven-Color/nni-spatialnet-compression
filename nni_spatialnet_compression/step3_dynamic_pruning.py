# Step 3: NNI动态剪枝实现
# ============================================================
# 本步骤演示如何使用NNI实现多种剪枝算法
# 包括：Level Pruner、L1/L2范数剪枝、FPGM、Movement Pruner等
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy
import nni
from nni.compression import TorchEvaluator
from nni.compression.pruning import (
    LevelPruner,      # 幅度剪枝（最基础）
    L1NormPruner,     # L1范数剪枝
    L2NormPruner,     # L2范数剪枝
    FPGMPruner,       # 基于几何中值的Filter剪枝
    MovementPruner,   # Movement剪枝（剪枝与训练结合）
    SlimPruner,       # 结构化稀疏剪枝
    TaylorPruner,     # Taylor展开剪枝
    LinearPruner,     # 线性调度剪枝
    AGPPruner         # 渐近剪枝
)
from nni.compression.utils import auto_set_denpendency_group_ids


def get_pruner_config_list(sparsity=0.5, op_types=['Conv2d', 'Linear']):
    """
    生成剪枝配置列表
    
    Args:
        sparsity: 目标稀疏度（0.5 = 剪枝50%的权重）
        op_types: 要剪枝的操作类型列表
        
    Returns:
        配置字典列表
    """
    config_list = [
        {
            'sparsity': sparsity,
            'op_types': op_types,
        }
    ]
    return config_list


def get_layer_specific_pruner_config(model: nn.Module, sparsity_per_layer: Dict[str, float]):
    """
    为特定层生成不同的剪枝配置
    
    Args:
        model: 待剪枝模型
        sparsity_per_layer: 层名到稀疏度的映射
        
    Returns:
        配置字典列表
    """
    config_list = []
    for name, module in model.named_modules():
        if name in sparsity_per_layer:
            config_list.append({
                'op_names': [name],
                'sparsity': sparsity_per_layer[name],
                'op_types': ['Conv2d', 'Linear']
            })
    return config_list


def getstructured_pruner_config(sparsity=0.5):
    """
    生成结构化剪枝配置（filter级别）
    
    Returns:
        配置字典列表
    """
    config_list = [
        {
            'sparsity': sparsity,
            'op_types': ['Conv2d'],
            'op_partial_names': ['stage1', 'stage2', 'stage3']  # 只剪枝这些阶段
        }
    ]
    return config_list


class PruningExperiment:
    """
    剪枝实验管理器
    用于执行不同的剪枝策略并比较结果
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, device: torch.device = None):
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cpu')
        self.results = {}
    
    def _get_fake_training_step(self):
        """创建用于NNI的伪造训练步骤"""
        def training_step(batch, model):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        return training_step
    
    def _get_fake_evaluating_func(self):
        """创建用于NNI的评估函数"""
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
        """评估原始模型的性能"""
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
    
    def prune_with_level_pruner(self, sparsity: float = 0.5) -> Tuple[nn.Module, Dict]:
        """
        使用Level Pruner进行幅度剪枝
        
        Level Pruner是最基础的剪枝方法，直接剪枝幅度最小的权重
        """
        print(f"\n{'='*60}")
        print(f"Level Pruner (稀疏度: {sparsity})")
        print(f"{'='*60}")
        
        # 复制模型以避免修改原始模型
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # 配置
        config_list = get_pruner_config_list(sparsity=sparsity)
        
        # 创建评估器
        training_step = self._get_fake_training_step()
        evaluating_func = self._get_fake_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer
        )
        
        # 创建剪枝器（不训练，只应用剪枝）
        pruner = LevelPruner(model, config_list, evaluator)
        
        # 执行压缩
        _, masks = pruner.compress()
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'masks': masks
        }
        self.results['level_pruner'] = result
        
        return model, result
    
    def prune_with_l1norm_pruner(self, sparsity: float = 0.5) -> Tuple[nn.Module, Dict]:
        """
        使用L1范数剪枝
        
        L1范数剪枝根据权重的L1范数来评估Filter/Channel的重要性，
        然后剪枝范数最小的Filter
        """
        print(f"\n{'='*60}")
        print(f"L1-Norm Pruner (稀疏度: {sparsity})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        config_list = get_pruner_config_list(sparsity=sparsity)
        
        training_step = self._get_fake_training_step()
        evaluating_func = self._get_fake_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer
        )
        
        pruner = L1NormPruner(model, config_list, evaluator)
        _, masks = pruner.compress()
        
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'masks': masks
        }
        self.results['l1norm_pruner'] = result
        
        return model, result
    
    def prune_with_fpgm_pruner(self, sparsity: float = 0.5) -> Tuple[nn.Module, Dict]:
        """
        使用FPGM（Filter Pruning via Geometric Median）剪枝
        
        FPGM选择最具可替代性的Filter进行剪枝
        """
        print(f"\n{'='*60}")
        print(f"FPGM Pruner (稀疏度: {sparsity})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        config_list = get_pruner_config_list(sparsity=sparsity)
        
        training_step = self._get_fake_training_step()
        evaluating_func = self._get_fake_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer
        )
        
        pruner = FPGMPruner(model, config_list, evaluator)
        _, masks = pruner.compress()
        
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'masks': masks
        }
        self.results['fpgm_pruner'] = result
        
        return model, result
    
    def prune_with_taylor_pruner(self, sparsity: float = 0.5, 
                                  training_steps: int = 100) -> Tuple[nn.Module, Dict]:
        """
        使用Taylor展开剪枝
        
        Taylor剪枝基于梯度信息来估计Filter的重要性
        需要进行训练来收集梯度信息
        """
        print(f"\n{'='*60}")
        print(f"Taylor Pruner (稀疏度: {sparsity})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        config_list = get_pruner_config_list(sparsity=sparsity)
        
        def training_step_with_max_steps(batch, model, optimizers, training_step,
                                          max_steps=None, max_epochs=None):
            """带max_steps控制的训练步骤"""
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            return loss
        
        training_step = lambda batch, model, optimizers=None, training_step=None, \
                              max_steps=None, max_epochs=None: training_step_with_max_steps(
                                  batch, model, optimizers, training_step, max_steps, max_epochs)
        
        evaluating_func = self._get_fake_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer
        )
        
        pruner = TaylorPruner(model, config_list, evaluator)
        _, masks = pruner.compress(max_steps=training_steps)
        
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'masks': masks,
            'training_steps': training_steps
        }
        self.results['taylor_pruner'] = result
        
        return model, result
    
    def prune_with_movement_pruner(self, sparsity: float = 0.5,
                                    warm_up_steps: int = 50,
                                    total_steps: int = 200) -> Tuple[nn.Module, Dict]:
        """
        使用Movement剪枝
        
        Movement剪枝是一种"边训练边剪枝"的方法，
        在训练过程中逐步增加稀疏度
        """
        print(f"\n{'='*60}")
        print(f"Movement Pruner (稀疏度: {sparsity})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        config_list = [
            {
                'sparsity': sparsity,
                'op_types': ['Conv2d', 'Linear'],
                'op_partial_names': ['classifier']
            }
        ]
        
        # 自动设置依赖组
        config_list = auto_set_denpendency_group_ids(model, config_list)
        
        def training_loop(model, optimizers, training_step, 
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            """模拟训练循环"""
            optimizer = optimizers
            total = max_steps if max_steps else 100
            
            model.train()
            batch_idx = 0
            for _ in range(total):
                for batch in self.train_loader:
                    if batch_idx >= total:
                        return
                    optimizer.zero_grad()
                    loss = training_step(batch, model)
                    loss.backward()
                    optimizer.step()
                    batch_idx += 1
        
        evaluating_func = self._get_fake_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer
        )
        
        pruner = MovementPruner(model, config_list, evaluator)
        _, masks = pruner.compress(max_steps=total_steps)
        
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'sparsity': sparsity,
            'accuracy': accuracy,
            'masks': masks,
            'warm_up_steps': warm_up_steps,
            'total_steps': total_steps
        }
        self.results['movement_pruner'] = result
        
        return model, result
    
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
        """比较所有剪枝方法的结果"""
        print("\n" + "=" * 60)
        print("剪枝方法比较")
        print("=" * 60)
        
        baseline_loss, baseline_acc = self.baseline_evaluation()
        print(f"\n基线模型: 损失={baseline_loss:.4f}, 准确率={baseline_acc:.2f}%")
        print("-" * 60)
        
        for name, result in self.results.items():
            acc = result['accuracy']
            sparsity = result['sparsity']
            print(f"{name}: 稀疏度={sparsity:.2f}, 准确率={acc:.2f}%")


def calculate_model_sparsity(model: nn.Module) -> Dict:
    """
    计算模型的稀疏度统计
    
    Returns:
        包含各层和总体稀疏度的字典
    """
    total_params = 0
    total_zeros = 0
    layer_stats = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total = param.numel()
            zeros = (param == 0).sum().item()
            sparsity = zeros / total
            
            total_params += total
            total_zeros += zeros
            
            layer_stats[name] = {
                'total': total,
                'zeros': zeros,
                'sparsity': sparsity
            }
    
    overall_sparsity = total_zeros / total_params if total_params > 0 else 0
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_params': total_params,
        'total_zeros': total_zeros,
        'layer_stats': layer_stats
    }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("NNI动态剪枝测试")
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
    
    # 创建剪枝实验
    experiment = PruningExperiment(model, train_loader, val_loader, device)
    
    # 评估基线
    print("\n评估基线模型...")
    baseline_loss, baseline_acc = experiment.baseline_evaluation()
    print(f"基线损失: {baseline_loss:.4f}")
    print(f"基线准确率: {baseline_acc:.2f}%")
    
    # 测试Level剪枝
    print("\n测试Level剪枝...")
    pruned_model, result = experiment.prune_with_level_pruner(sparsity=0.3)
    print(f"Level剪枝后准确率: {result['accuracy']:.2f}%")
    
    # 测试L1-Norm剪枝
    print("\n测试L1-Norm剪枝...")
    pruned_model, result = experiment.prune_with_l1norm_pruner(sparsity=0.3)
    print(f"L1-Norm剪枝后准确率: {result['accuracy']:.2f}%")
    
    # 比较结果
    experiment.compare_results()
    
    # 计算稀疏度
    print("\n计算剪枝后模型稀疏度...")
    stats = calculate_model_sparsity(pruned_model)
    print(f"总体稀疏度: {stats['overall_sparsity']:.2%}")
    print(f"零值参数: {stats['total_zeros']:,} / {stats['total_params']:,}")
    
    print("\n剪枝测试完成!")
