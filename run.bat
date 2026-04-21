"""
Windows 批处理版本 run.bat
==========================
用于 Windows 环境直接双击运行
"""
@echo off
setlocal enabledelayedexpansion

set "PYTHON=python"

echo ========================================
echo 模型压缩实验 - 一键运行 (Windows)
echo ========================================

:: 检查 Python
!PYTHON! -c "import torch" 2>nul
if errorlevel 1 (
    echo [ERROR] 未安装 PyTorch，请运行: pip install torch torchvision
    pause
    exit /b 1
)

:: 检查 PyYAML
!PYTHON! -c "import yaml" 2>nul
if errorlevel 1 (
    echo [INFO] 安装 PyYAML...
    pip install pyyaml
)

echo [INFO] 开始运行...

:: 训练
echo.
echo [INFO] ========== 训练阶段 ==========
!PYTHON! train.py
if errorlevel 1 (
    echo [ERROR] 训练失败
    pause
    exit /b 1
)

:: 剪枝
echo.
echo [INFO] ========== 剪枝阶段 ==========
!PYTHON! prune.py
if errorlevel 1 (
    echo [ERROR] 剪枝失败
    pause
    exit /b 1
)

:: 量化
echo.
echo [INFO] ========== 量化阶段 ==========
!PYTHON! quantize.py
if errorlevel 1 (
    echo [ERROR] 量化失败
    pause
    exit /b 1
)

:: 汇总
echo.
echo [INFO] ========== 结果汇总 ==========
!PYTHON! summary.py

echo.
echo ========================================
echo [OK] 所有阶段完成!
echo ========================================
echo 结果目录: results\
echo 查看汇总: results\summary.json
pause
