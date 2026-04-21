# MNIST 数据集

## 数据集说明

MNIST 是一个经典的手写数字识别数据集，常用于机器学习入门和模型压缩实验。

## 数据信息

| 属性 | 值 |
|------|-----|
| 图像尺寸 | 28×28 灰度 |
| 训练集 | 60,000 张 |
| 测试集 | 10,000 张 |
| 类别数 | 10 (数字 0-9) |
| 数据来源 | http://yann.lecun.com/exdb/mnist/ |

## 文件结构

```
datasets/mnist/
├── raw/           # 原始 MNIST 数据文件 (gzip)
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
└── README.md
```

## 使用方式

```python
from torchvision import datasets

# PyTorch 自动下载
train_ds = datasets.MNIST(root='./data', train=True, download=True)
test_ds = datasets.MNIST(root='./data', train=False, download=True)
```

## 在本项目中的使用

所有压缩工具 (NNI, Intel INC, Amazon AMM, PyTorch Native) 均使用 MNIST 作为实验数据集，确保结果可对比。

## 参考

- 官网: http://yann.lecun.com/exdb/mnist/
- PyTorch: https://pytorch.org/vision/stable/datasets.html#mnist