#!/bin/bash
# run_resample_aug.sh - 构建增强训练数据（原始 + 旋转/缩放 + 域降质）

PROJECT_ROOT="/root/Projects/meteorite-identification"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "构建增强训练数据（含域降质）"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

python src/data_processing_modules/resample_with_degraded.py

echo ""
echo "=========================================="
echo "完成!"
echo "结束时间: $(date)"
echo "=========================================="
