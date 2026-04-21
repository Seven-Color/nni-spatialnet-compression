"""
模型压缩实验 - 主入口
====================
直接执行所有操作，模块化但不过度封装

使用方法:
    python run.py                  # 全部
    python run.py train            # 仅训练
    python run.py prune            # 仅剪枝
    python run.py quantize         # 仅量化
"""

import os
import sys
import json
import argparse

# 加载配置
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# 导入模块
from model import build_model, count_params
from train import run_train
from prune import run_prune
from quantize import run_quantize
from summary import run_summary


def main():
    parser = argparse.ArgumentParser(description="模型压缩实验")
    parser.add_argument("stage", nargs="?", choices=["all", "train", "prune", "quantize", "summary"], 
                        default="all", help="执行阶段 (默认: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型压缩实验")
    print("=" * 60)
    print(f"配置: {config['model']['layers']}层, 通道={config['model']['channels']}")
    print(f"剪枝: {config['prune']['enable']}, 稀疏度={config['prune']['sparsity']}")
    print(f"量化: {config['quantize']['enable']}, 方法={config['quantize']['method']}")
    print("=" * 60)
    
    results = {}
    
    if args.stage in ["all", "train"]:
        results["train"] = run_train(config)
    
    if args.stage in ["all", "prune"]:
        results["prune"] = run_prune(config)
    
    if args.stage in ["all", "quantize"]:
        results["quantize"] = run_quantize(config)
    
    if args.stage in ["all", "summary"]:
        results["summary"] = run_summary(config)
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("=" * 60)
    print(f"结果目录: {config['output']['dir']}/")


if __name__ == "__main__":
    main()