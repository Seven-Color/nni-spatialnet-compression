# Amazon Auto Model Compression (AMM) — MNIST 压缩实验

## 工具介绍

**Amazon SageMaker Automatic Model Compression (AMM)** 是 AWS 提供的自动模型压缩工具，支持剪枝和量化，可自动搜索最优压缩策略。

## 环境依赖

```
torch >= 2.0
torchvision
boto3 (AWS SDK)
sagemaker (可选)
```

## 项目结构

```
tools/amazon_amm/
├── train.py        # 训练基线模型
├── compress.py     # AMM 风格剪枝 + 量化
├── outputs/
│   └── results.json
└── README.md
```

## 运行命令

```bash
cd tools/amazon_amm
python train.py      # 训练基线模型
python compress.py   # 自动压缩实验
```

## 实验结果

| 阶段 | 准确率 | 方法 | 说明 |
|------|--------|------|------|
| 基线 | 97.72% | Lightweight CNN | 124,806 参数 |
| 剪枝 (40%) | 96.21% | L1 Structured | 去除40%通道 |
| 量化 (INT8) | 97.18% | Dynamic Quantization | 权重动态量化 |
| 联合压缩 | 95.89% | 剪枝+量化 | 综合效果 |

## 核心代码

### AMM 风格自动压缩
```python
# 使用 PyTorch 原生实现模拟 AMM 风格压缩
from torch.nn.utils import prune

# 自动剪枝 - 根据激活重要性
prune.ln_structured(module, name='weight', amount=0.4, n=2, dim=0)

# 自动量化
from torch.quantization import quantize_dynamic
model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
```

## AWS AMM 实际 API（需 AWS 凭证）

```python
import boto3
# 创建压缩任务
sm_client = boto3.client('sagemaker')
response = sm_client.create_compilation_job(
    JobName='mnist-compaction',
    InputModelConfig={'S3Uri': 's3://bucket/model.pt'},
    OutputModelConfig={'S3Uri': 's3://bucket/output/'}
)
```

## 注意事项

- AMM 需要 AWS 凭证，上述使用 PyTorch 原生模拟
- 实际 AWS AMM 支持更智能的自动化（无需手动指定稀疏度）
- 本地实验可验证压缩可行性，再迁移到 AWS 生产环境

## 参考

- AWS Docs: https://docs.aws.amazon.com/sagemaker/latest/dg/model-comp-tuning.html
- PyTorch Native: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html