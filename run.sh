#!/bin/bash
# ============================================================
# 模型压缩实验一键运行脚本
# ============================================================
# 用法:
#   bash run.sh           # 运行所有阶段
#   bash run.sh train     # 仅训练
#   bash run.sh prune     # 仅剪枝
#   bash run.sh quantize  # 仅量化
#   bash run.sh summary   # 仅汇总
# ============================================================

set -e  # 遇到错误立即退出

# 配置
CONFIG_FILE="config.yaml"
PYTHON="${PYTHON:-python}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_deps() {
    log_info "检查 Python 依赖..."
    
    # 检查 PyTorch
    if ! $PYTHON -c "import torch" 2>/dev/null; then
        log_error "未安装 PyTorch，请运行: pip install torch torchvision"
        exit 1
    fi
    
    # 检查 PyYAML
    if ! $PYTHON -c "import yaml" 2>/dev/null; then
        log_info "安装 PyYAML..."
        pip install pyyaml
    fi
    
    # 检查 torchvision
    if ! $PYTHON -c "import torchvision" 2>/dev/null; then
        log_error "未安装 torchvision，请运行: pip install torchvision"
        exit 1
    fi
    
    log_info "依赖检查完成"
}

# 运行指定阶段
run_train() {
    log_info "========== 训练阶段 =========="
    $PYTHON train.py
}

run_prune() {
    log_info "========== 剪枝阶段 =========="
    $PYTHON prune.py
}

run_quantize() {
    log_info "========== 量化阶段 =========="
    $PYTHON quantize.py
}

run_summary() {
    log_info "========== 结果汇总 =========="
    $PYTHON summary.py
}

# 主流程
run_all() {
    log_info "========================================"
    log_info "模型压缩实验 - 一键运行"
    log_info "========================================"
    
    check_deps
    run_train
    run_prune
    run_quantize
    run_summary
    
    log_info "========================================"
    log_info "✅ 所有阶段完成!"
    log_info "========================================"
    log_info "结果目录: results/"
    log_info "查看汇总: results/summary.json"
}

# 帮助
show_help() {
    echo "用法: bash run.sh [命令]"
    echo ""
    echo "命令:"
    echo "  (无参数)   运行所有阶段 (训练 → 剪枝 → 量化 → 汇总)"
    echo "  train      仅训练"
    echo "  prune      仅剪枝"
    echo "  quantize   仅量化"
    echo "  summary    仅汇总"
    echo "  help       显示此帮助"
    echo ""
    echo "配置:"
    echo "  修改 config.yaml 调整模型和实验参数"
}

# 主入口
case "${1:-all}" in
    train)
        check_deps
        run_train
        ;;
    prune)
        check_deps
        run_prune
        ;;
    quantize)
        check_deps
        run_quantize
        ;;
    summary)
        run_summary
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "未知命令: $1"
        show_help
        exit 1
        ;;
esac
