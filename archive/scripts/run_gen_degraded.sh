#!/bin/bash
# run_gen_degraded.sh - 生成域降质增强图像

PROJECT_ROOT="/root/Projects/meteorite-identification"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "生成域降质增强图像"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

python -m src.data_processing_modules.gen_degraded_data "$@"

echo ""
echo "=========================================="
echo "完成!"
echo "结束时间: $(date)"
echo "=========================================="
