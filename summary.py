"""结果汇总模块"""
import os
import json
import yaml
from pathlib import Path


def load_json(path):
    """加载 JSON 文件"""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def summarize(config):
    """生成实验汇总"""
    print("\n" + "="*60)
    print("📊 实验结果汇总")
    print("="*60)
    
    output_dir = config['output']['dir']
    
    # 加载各阶段结果
    train_results = load_json(os.path.join(output_dir, "train_results.json"))
    prune_results = load_json(os.path.join(output_dir, "prune_results.json"))
    quant_results = load_json(os.path.join(output_dir, "quantize_results.json"))
    
    # 打印汇总表格
    print(f"\n模型: {train_results.get('model_name', 'N/A')}")
    print(f"参数量: {train_results.get('params', 'N/A'):,}")
    print()
    print("┌────────────┬────────────┬────────────┐")
    print("│ 阶段       │ 准确率     │ 变化       │")
    print("├────────────┼────────────┼────────────┤")
    
    print(f"│ 基线       │ {train_results.get('accuracy', 'N/A'):>8.2f}% │            │")
    
    if prune_results:
        drop = prune_results.get('accuracy_drop', 0)
        print(f"│ 剪枝       │ {prune_results['accuracy']:>8.2f}% │ {drop:>+8.2f}% │")
    
    if quant_results:
        drop = quant_results.get('accuracy_drop', 0)
        print(f"│ 量化       │ {quant_results['accuracy']:>8.2f}% │ {drop:>+8.2f}% │")
    
    print("└────────────┴────────────┴────────────┘")
    
    # 保存汇总
    summary = {
        "config": config,
        "results": {
            "train": train_results,
            "prune": prune_results,
            "quantize": quant_results
        }
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 汇总已保存: {summary_path}")
    
    return summary


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    summarize(config)
