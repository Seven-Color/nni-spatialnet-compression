# MNIST SpatialNet 压缩演示 - 分步测试

## 文件说明

| 文件 | 说明 | 运行命令 |
|------|------|----------|
| step1_spatialnet_mnist.py | 模型定义（4层SpatialNet） | - |
| step1_train.py | 训练模型 | `python step1_train.py` |
| step2_prune.py | Level剪枝 | `python step2_prune.py` |
| step3_quantize.py | QAT量化 | `python step3_quantize.py` |

## 分步测试流程

### Step 1: 训练模型
```bash
python step1_train.py
```
输出:
- mnist_spatialnet_trained.pth (训练好的模型)
- 预期准确率: ~98%

### Step 2: 剪枝
```bash
python step2_prune.py
```
输出:
- mnist_spatialnet_pruned.pth (剪枝后的模型)
- 预期准确率变化: 约 -0.5% ~ -2%

### Step 3: 量化
```bash
python step3_quantize.py
```
输出:
- mnist_spatialnet_quantized.pth (量化后的模型)
- 预期准确率变化: 约 -0.5% ~ +0.5% (通过QAT训练恢复)

## 模型结构

4层SpatialNet用于MNIST分类:
```
Layer1: Conv(1→32) + BN + ReLU + Attention + Pool
Layer2: Conv(32→64) + BN + ReLU + Attention + Pool  
Layer3: Conv(64→128) + BN + ReLU + Attention + Pool
Layer4: Conv(128→256) + BN + ReLU + Attention
GlobalAvgPool → Linear(256→10)
```

参数量: 391,854

## NNI压缩方法

### 剪枝 (Pruning)
- **LevelPruner**: 简单稀疏度剪枝，按比例将权重置零
- **L1NormPruner**: 基于L1范数的重要性剪枝
- **AGPPruner**: 自动化渐进式剪枝

### 量化 (Quantization)
- **PtqQuantizer**: 训练后量化，需要校准数据
- **QATQuantizer**: 量化感知训练，在训练中模拟量化效果

## 预期结果

| 阶段 | 准确率 | 压缩比 |
|------|--------|--------|
| 训练后(基线) | ~98% | 1x |
| 剪枝(50%) | ~96-97% | ~1x (参数还在) |
| QAT量化(8-bit) | ~97-98% | 4x (理论) |

注意: 剪枝只是将权重置零，实际模型大小不变；量化将权重从32-bit减少到8-bit，理论压缩4x。