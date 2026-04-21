# 实验结果汇总

本文件夹汇总所有模型压缩工具的实验结果。

## 实验结果文件

| 文件 | 说明 |
|------|------|
| `工具对比分析报告.md` | 完整的对比分析报告 |
| `汇报_模型压缩工具对比分析_v20260421.pptx` | 综合汇报 PPT |

## 实验数据来源

| 工具 | results.json 位置 |
|------|------------------|
| NNI | `../nni_spatialnet_compression/` |
| Intel INC | `../tools/intel_neural_compressor/outputs/results.json` |
| Amazon AMM | `../tools/amazon_amm/outputs/results.json` |
| PyTorch Native | `../tools/pytorch_native_compression/outputs/results.json` |

## 快速查看实验结果

```bash
# 查看所有工具结果
cat ../tools/*/outputs/results.json
```