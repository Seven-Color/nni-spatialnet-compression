# TensorRT Compression — MNIST 压缩实验

## 工具介绍

**TensorRT** 是 NVIDIA 的高性能推理引擎，支持模型优化、量化（INT8/FP16）、层融合等，可显著加速 GPU 推理。本实验使用 PyTorch 训练 + TensorRT 优化流水线。

## 环境依赖

```
torch >= 2.0
torchvision
tensorrt (pip install tensorrt)  # 需要 NVIDIA GPU/CUDA
```

## 项目结构

```
tools/tensorrt_compression/
├── train.py        # 训练基线模型
├── export.py       # 导出 ONNX
├── optimize.py     # TensorRT 优化
├── outputs/
│   └── results.json
└── README.md
```

## 运行命令

```bash
cd tools/tensorrt_compression
python train.py       # 训练基线
python export.py      # 导出 ONNX
python optimize.py    # TensorRT INT8 优化
```

## 实验结果

| 阶段 | 准确率 | 方法 | 说明 |
|------|--------|------|------|
| 基线 | 97.85% | PyTorch CNN | FP32 |
| 剪枝 (50%) | 96.42% | Prune + TensorRT | 稀疏优化 |
| 量化 (INT8) | 97.31% | TensorRT INT8 | 4x 压缩 |
| 联合压缩 | 96.18% | 剪枝+INT8 | 最优推理速度 |

## 核心代码

### PyTorch → ONNX 导出
```python
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
```

### TensorRT INT8 优化
```python
import tensorrt as trt

# 构建 TensorRT engine
builder = trt.Builder()
network = builder.create_network()
parser = trt.OnnxParser(network)
parser.parse_from_file("model.onnx")

# INT8 量化
builder.int8_mode = True
builder.int8_calibrator = calibrator

# 构建 engine
engine = builder.build_cuda_engine(network)
```

## TensorRT 优化技术

| 技术 | 效果 |
|------|------|
| FP16 | 2x 加速，精度损失 < 0.5% |
| INT8 | 4x 加速，需要校准数据 |
| Layer Fusion | 合并卷积+BN+ReLU，减少内存访问 |
| Kernel Auto-Tuning | 选择最优 CUDA kernel |
| DLA | 边缘设备加速（可选） |

## 注意事项

- TensorRT 需要 NVIDIA GPU + CUDA 环境
- INT8 量化需要校准数据集（MNIST 选 500 张）
- 优化后模型仅支持 NVIDIA GPU 推理

## 参考

- TensorRT Docs: https://docs.nvidia.com/deeplearning/tensorrt/
- PyTorch → TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/