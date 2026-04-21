# 模型压缩实验

基于 PyTorch 的模型剪枝和量化实验，支持 MNIST 数据集。

## 环境配置

```bash
# 创建并激活环境
conda create -n model_compression python=3.10 -y
conda activate model_compression

# 安装依赖
pip install torch torchvision pyyaml
```

## 使用方法

```bash
python run.py                  # 全部: 训练 → 剪枝 → 量化 → 汇总
python run.py train            # 仅训练
python run.py prune            # 仅剪枝
python run.py quantize         # 仅量化
python run.py summary          # 仅汇总
```

或单独运行各模块：

```bash
python train.py
python prune.py
python quantize.py
python summary.py
```

## 配置文件

编辑 `config.yaml` 调整参数。

### 放大模型

```yaml
# 2层 (默认，快速验证)
model:
  layers: 2
  channels: [16, 32]
  fc_hidden: 128

# 3层
model:
  layers: 3
  channels: [16, 32, 64]
  fc_hidden: 256

# 4层 (更大)
model:
  layers: 4
  channels: [32, 64, 128, 256]
  fc_hidden: 512
```

### 调整训练

```yaml
train:
  epochs: 2        # 可增加到 5, 10 等
  batch_size: 256
  lr: 0.001
```

### 调整剪枝

```yaml
prune:
  enable: true
  sparsity: 0.5    # 0.3 = 剪掉30%, 0.7 = 剪掉70%
  method: "l1_unstructured"  # l1_unstructured, random_unstructured, ln_structured
```

### 调整量化

```yaml
quantize:
  enable: true
  method: "dynamic_int8"    # dynamic_int8, static_int8, qat
```

## 输出结果

```
results/
├── baseline.pt             # 基线模型
├── pruned.pt               # 剪枝后模型
├── quantized.pt            # 量化后模型
├── train_results.json
├── prune_results.json
├── quantize_results.json
└── summary.json            # 实验汇总
```

## 项目结构

```
├── config.yaml     # 实验配置
├── model.py        # 模型定义
├── train.py        # 训练
├── prune.py        # 剪枝
├── quantize.py     # 量化
├── summary.py      # 汇总
├── run.py          # 主入口
└── results/        # 输出目录
```

## 依赖

- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- PyYAML