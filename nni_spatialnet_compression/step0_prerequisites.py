# Step 0: 环境检查与依赖安装
# ============================================================
# 本步骤确保所有必要的依赖已正确安装
# ============================================================

import sys
print(f"Python版本: {sys.version}")

# 检查PyTorch
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
except ImportError as e:
    print(f"PyTorch未安装: {e}")
    print("请运行: pip install torch torchvision")

# 检查NNI
try:
    import nni
    print(f"NNI版本: {nni.__version__}")
except ImportError as e:
    print(f"NNI未安装: {e}")
    print("请运行: pip install nni")

# 检查NNI压缩模块
try:
    from nni.compression import TorchEvaluator
    from nni.compression.pruning import LevelPruner, L1NormPruner, L2NormPruner, FPGMPruner
    from nni.compression.quantization import QATQuantizer, PtqQuantizer, DoReFaQuantizer
    print("NNI压缩模块导入成功!")
except ImportError as e:
    print(f"NNI压缩模块导入失败: {e}")

print("\n环境检查完成!")
