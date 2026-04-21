# PyTorch Native Model Compression — MNIST 压缩实验

## 工具介绍

**PyTorch Native Compression** 是 PyTorch 内置的模型压缩能力，包括 `torch.nn.utils.prune`（剪枝）和 `torch.quantization`（量化）。无需额外安装，PyTorch 自带。

## 环境依赖

```
torch >= 1.7
torchvision
```

## 项目结构

```
tools/pytorch_native_compression/
├── train.py        # 训练基线模型
├── prune.py        # 多种剪枝方法
├── quantize.py     # PTQ / QAT 量化
├── combined.py     # 联合压缩
├── quick.py        # 快速实验脚本
├── outputs/
│   └── results.json
└── README.md
```

## 运行命令

```bash
cd tools/pytorch_native_compression
python train.py       # 训练基线模型
python prune.py       # 剪枝实验
python quantize.py    # 量化实验
python combined.py    # 联合压缩
python quick.py       # 一键快速实验
```

## 实验结果

| 阶段 | 准确率 | 方法 | 说明 |
|------|--------|------|------|
| 基线 | 97.85% | 2层 CNN | 25,874 参数 |
| 剪枝 (50%) | 96.42% | L1 Unstructured | 去除50%权重 |
| 量化 (INT8) | 97.31% | Dynamic Quantization | 权重INT8 |
| 联合压缩 | 96.18% | 剪枝+量化 | 综合效果 |

## 核心代码

### 剪枝方法
```python
from torch.nn.utils import prune

# Unstructured Pruning（逐元素剪枝）
prune.l1_unstructured(module, name='weight', amount=0.5)  # L1重要性
prune.random_unstructured(module, name='weight', amount=0.5)  # 随机

# Structured Pruning（按通道/神经元剪枝）
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)  # Ln范数
prune.random_structured(module, name='weight', amount=0.3, dim=0)  # 随机通道
```

### 量化方法
```python
# 动态量化（权重 int8，激活动态量化）
from torch.quantization import quantize_dynamic
model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

# 静态量化 PTQ（需校准）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# 校准
torch.quantization.convert(model, inplace=True)

# QAT 量化感知训练
torch.quantization.enable_observer(model)
torch.quantization.enable_fake_quant(model)
```

## 剪枝 vs 量化区别

| 特性 | 剪枝 (Pruning) | 量化 (Quantization) |
|------|-----------------|---------------------|
| 压缩方式 | 将权重置零 | 降低权重精度 |
| 文件大小 | 不变（稀疏矩阵存储） | 减小（4x for INT8） |
| 推理加速 | 需要稀疏计算库支持 | 直接加速 |
| 精度损失 | 中等 | 较小（QAT 可恢复） |

## 自动化程度

- **剪枝**: 半自动（需指定 sparsity）
- **量化**: 半自动（需指定 dtype）
- **推荐**: 使用 NNI/INC 等框架实现全自动搜索

## 参考

- PyTorch Pruning: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html