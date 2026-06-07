#!/bin/bash
# run_predict_baseline.sh - 基线模型对 test_images-v2 推理

PROJECT_ROOT="/root/Projects/meteorite-identification"
SCRIPT_DIR="${PROJECT_ROOT}/src/testing"

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "基线模型推理：test_images-v2"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

python gen_submission_baseline.py

echo ""
echo "=========================================="
echo "完成!"
echo "结束时间: $(date)"
echo "=========================================="
