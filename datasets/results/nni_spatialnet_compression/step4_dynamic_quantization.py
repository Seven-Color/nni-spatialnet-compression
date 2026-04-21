# Step 4: NNI动态量化实现
# ============================================================
# 本步骤演示如何使用NNI实现多种量化算法
# 包括：PTQ、QAT、DoReFa、LSQ等
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy
import nni
from nni.compression import TorchEvaluator
from nni.compression.quantization import (
    QATQuantizer,      # Quantization-Aware Training
    PtqQuantizer,      # Post-Training Quantization
    DoReFaQuantizer,   # DoReFa梯度量化
    LsqQuantizer,      # Learned Step Size Quantization
    BNNQuantizer,      # Binary Neural Network
    LsqPlusQuantizer,  # LSQ+
)


def get_quantizer_config_list(quant_types=['weight', 'output'],
                               quant_bits=8,
                               op_types=['Conv2d', 'Linear']):
    """
    生成量化配置列表
    
    Args:
        quant_types: 要量化的类型 ['weight', 'output', 'input']
        quant_bits: 量化位数
        op_types: 要量化的操作类型
        
    Returns:
        配置字典列表
    """
    config_list = []
    
    if 'weight' in quant_types:
        config_list.append({
            'op_types': op_types,
            'weight_quantizer': {
                'type': 'PerTensorAffine',
                'quant_bits': quant_bits,
            }
        })
    
    if 'output' in quant_types:
        config_list.append({
            'op_types': op_types,
            'output_quantizer': {
                'type': 'PerTensorAffine',
                'quant_bits': quant_bits,
            }
        })
    
    return config_list


def get_qat_config_list(quant_bits=8, op_types=['Conv2d', 'Linear']):
    """
    生成QAT量化配置
    
    Args:
        quant_bits: 量化位数
        op_types: 要量化的操作类型
        
    Returns:
        配置字典列表
    """
    config_list = [
        {
            'op_types': op_types,
            'quant_types': ['weight', 'output'],
            'quant_bits': quant_bits,
            'quant_scheme': 'affine',  # 或 'symmetric'
        }
    ]
    return config_list


def get_ptq_config_list(calibration_config: Dict = None):
    """
    生成PTQ（训练后量化）配置
    
    Args:
        calibration_config: 校准配置，包含数据加载器等信息
        
    Returns:
        配置字典列表
    """
    if calibration_config is None:
        calibration_config = {}
    
    config_list = [
        {
            'op_types': ['Conv2d', 'Linear'],
            'quant_types': ['weight', 'output'],
            'quant_bits': 8,
        }
    ]
    return config_list


class QuantizationExperiment:
    """
    量化实验管理器
    用于执行不同的量化策略并比较结果
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device = None):
        self.original_model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cpu')
        self.results = {}
    
    def _get_training_step(self, device):
        """创建训练步骤"""
        def training_step(batch, model):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            return loss
        return training_step
    
    def _get_full_training_loop(self, max_epochs=5):
        """创建完整训练循环"""
        def training_loop(model, optimizers, training_step,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            epochs = max_epochs if max_epochs else max_epochs
            optimizer = optimizers
            
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
    
    def quantize_with_ptq(self, quant_bits: int = 8) -> Tuple[nn.Module, Dict]:
        """
        使用PTQ（训练后量化）
        
        PTQ是最简单的量化方法，不需要重新训练模型
        只需要一小部分数据进行校准
        """
        print(f"\n{'='*60}")
        print(f"PTQ Quantizer (量化位数: {quant_bits})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # PTQ配置
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_types': ['weight', 'output'],
                'quant_bits': quant_bits,
            }
        ]
        
        # 评估器
        training_step = self._get_training_step(self.device)
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_step,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = PtqQuantizer(model, config_list, evaluator)
        
        # 执行压缩（PTQ只需要校准，不需要训练）
        _, calibration_config = quantizer.compress()
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'calibration_config': calibration_config
        }
        self.results['ptq_quantizer'] = result
        
        return model, result
    
    def quantize_with_qat(self, quant_bits: int = 8,
                          quant_start_step: int = 500,
                          max_epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        使用QAT（量化感知训练）
        
        QAT在训练过程中模拟量化效果，需要较长训练时间
        但能获得更好的量化精度
        """
        print(f"\n{'='*60}")
        print(f"QAT Quantizer (量化位数: {quant_bits})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # QAT配置
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_types': ['weight', 'output'],
                'quant_bits': quant_bits,
            }
        ]
        
        # 训练循环
        def training_loop(model, optimizers, training_step,
                         lr_schedulers=None, max_steps=None, max_epochs=None):
            optimizer = optimizers
            epochs = max_epochs if max_epochs else max_epochs
            steps_per_epoch = len(self.train_loader)
            total_steps = epochs * steps_per_epoch
            
            model.train()
            step = 0
            for epoch in range(epochs):
                for batch in self.train_loader:
                    if step >= total_steps:
                        return
                    optimizer.zero_grad()
                    loss = training_step(batch, model)
                    loss.backward()
                    optimizer.step()
                    step += 1
        
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = QATQuantizer(
            model, config_list, evaluator,
            quant_start_step=quant_start_step
        )
        
        # 执行压缩
        _, calibration_config = quantizer.compress(max_epochs=max_epochs)
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'quant_start_step': quant_start_step,
            'max_epochs': max_epochs,
            'calibration_config': calibration_config
        }
        self.results['qat_quantizer'] = result
        
        return model, result
    
    def quantize_with_lsq(self, quant_bits: int = 8,
                          max_epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        使用LSQ（ Learned Step Size Quantization）
        
        LSQ通过学习每个量化层的步长来获得更好的量化精度
        """
        print(f"\n{'='*60}")
        print(f"LSQ Quantizer (量化位数: {quant_bits})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # LSQ配置
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_types': ['weight', 'output'],
                'quant_bits': quant_bits,
            }
        ]
        
        # 训练循环
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
        
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = LsqQuantizer(model, config_list, evaluator)
        
        # 执行压缩
        _, calibration_config = quantizer.compress(max_epochs=max_epochs)
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'max_epochs': max_epochs,
            'calibration_config': calibration_config
        }
        self.results['lsq_quantizer'] = result
        
        return model, result
    
    def quantize_with_dorefa(self, quant_bits: int = 8,
                             max_epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        使用DoReFa量化
        
        DoReFa是早期的量化训练方法之一
        使用分层量化的方式
        """
        print(f"\n{'='*60}")
        print(f"DoReFa Quantizer (量化位数: {quant_bits})")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # DoReFa配置
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_types': ['weight', 'output'],
                'quant_bits': quant_bits,
            }
        ]
        
        # 训练循环
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
        
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = DoReFaQuantizer(model, config_list, evaluator)
        
        # 执行压缩
        _, calibration_config = quantizer.compress(max_epochs=max_epochs)
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'quant_bits': quant_bits,
            'accuracy': accuracy,
            'max_epochs': max_epochs,
            'calibration_config': calibration_config
        }
        self.results['dorefa_quantizer'] = result
        
        return model, result
    
    def quantize_with_bnn(self, max_epochs: int = 5) -> Tuple[nn.Module, Dict]:
        """
        使用BNN（二值化神经网络）
        
        BNN将权重和激活二值化（1bit）
        能极大减少模型大小和计算量
        """
        print(f"\n{'='*60}")
        print(f"BNN Quantizer (二值化)")
        print(f"{'='*60}")
        
        model = copy.deepcopy(self.original_model).to(self.device)
        
        # BNN配置
        config_list = [
            {
                'op_types': ['Conv2d', 'Linear'],
                'quant_types': ['weight', 'output'],
                'quant_bits': 1,  # BNN使用1bit
            }
        ]
        
        # 训练循环
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
        
        evaluating_func = self._get_evaluating_func()
        
        traced_optimizer = nni.trace(torch.optim.Adam)(model.parameters(), lr=0.001)
        evaluator = TorchEvaluator(
            training_func=training_loop,
            optimizers=traced_optimizer,
            evaluating_func=evaluating_func
        )
        
        # 创建量化器
        quantizer = BNNQuantizer(model, config_list, evaluator)
        
        # 执行压缩
        _, calibration_config = quantizer.compress(max_epochs=max_epochs)
        
        # 评估
        _, accuracy = self._evaluate_model(model)
        
        result = {
            'quant_bits': 1,
            'accuracy': accuracy,
            'max_epochs': max_epochs,
            'calibration_config': calibration_config
        }
        self.results['bnn_quantizer'] = result
        
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
        """比较所有量化方法的结果"""
        print("\n" + "=" * 60)
        print("量化方法比较")
        print("=" * 60)
        
        baseline_loss, baseline_acc = self.baseline_evaluation()
        print(f"\n基线模型: 损失={baseline_loss:.4f}, 准确率={baseline_acc:.2f}%")
        print("-" * 60)
        
        for name, result in self.results.items():
            acc = result['accuracy']
            bits = result.get('quant_bits', 'N/A')
            print(f"{name}: {bits}bit, 准确率={acc:.2f}%")


def analyze_quantization_bits(model: nn.Module) -> Dict:
    """
    分析模型各层的量化位数
    
    Returns:
        包含各层量化信息的字典
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_info[name] = {
                'type': type(module).__name__,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size if hasattr(module, 'kernel_size') else 'N/A',
            }
    
    return layer_info


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("NNI动态量化测试")
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
    
    # 创建量化实验
    experiment = QuantizationExperiment(model, train_loader, val_loader, device)
    
    # 评估基线
    print("\n评估基线模型...")
    baseline_loss, baseline_acc = experiment.baseline_evaluation()
    print(f"基线损失: {baseline_loss:.4f}")
    print(f"基线准确率: {baseline_acc:.2f}%")
    
    # 测试PTQ量化
    print("\n测试PTQ量化...")
    quantized_model, result = experiment.quantize_with_ptq(quant_bits=8)
    print(f"PTQ量化后准确率: {result['accuracy']:.2f}%")
    
    # 比较结果
    experiment.compare_results()
    
    print("\n量化测试完成!")
