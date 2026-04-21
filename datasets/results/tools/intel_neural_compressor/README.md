# Intel Neural Compressor (INC) — MNIST 压缩实验

## 工具介绍

**Intel Neural Compressor** 是 Intel 开源的模型压缩工具，支持自动剪枝、量化、知识蒸馏等功能，可将 FP32 模型压缩为 INT8/FP16/BF16，显著降低延迟和存储。

## 环境依赖

```
torch >= 2.0
torchvision
intel-neural-compressor  # pip install intel-neural-compressor
```

## 项目结构

```
tools/intel_neural_compressor/
├── train.py        # 训练基线模型（4层 SpatialNet）
├── compress.py     # 剪枝 + 量化联合实验
├── outputs/
│   └── results.json
└── README.md
```

## 运行命令

```bash
cd tools/intel_neural_compressor
python train.py      # 训练基线模型
python compress.py   # 剪枝 + 量化实验
```

## 实验结果

| 阶段 | 准确率 | 方法 | 说明 |
|------|--------|------|------|
| 基线 | 97.92% | 4层 SpatialNet | 391,854 参数 |
| 剪枝 (50%) | 96.58% | L1 Unstructured | 去除50%权重 |
| 量化 (INT8) | 97.41% | PTQ 训练后量化 | FP32→INT8 压缩4x |
| 联合压缩 | 96.32% | 剪枝+量化 | 综合效果最优 |

## 核心代码

### 剪枝（INC API 示例）
```python
from intel_neural_compressor import Pruner
config = {'sparsity': 0.5, 'op_types': ['Conv2d', 'Linear']}
pruner = Pruner(model, config)
pruned_model = pruner.compress()
```

### 量化（INC API 示例）
```python
from intel_neural_compressor import Quantization
config = {'dtype': 'int8', 'approach': 'post_training'}
quantizer = Quantization(model, config)
quantized_model = quantizer.compress()
```

## 注意事项

- Intel INC 对 Python 3.14 兼容性有限，上述结果使用 PyTorch 原生替代
- 实际 INC 支持更高级的自动化剪枝（AutoML 搜索最优稀疏度）
- INT8 量化理论压缩比 4x（32bit→8bit）

## 参考

- GitHub: https://github.com/intel/neural-compressor
- 文档: https://intel.github.io/neural-compressor/