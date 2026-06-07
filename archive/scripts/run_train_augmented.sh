#!/bin/bash
# run_train_augmented.sh - 增强版训练（TokenMixUp + 多尺度 + 域降质数据）

PROJECT_ROOT="/root/Projects/meteorite-identification"
SCRIPT_DIR="${PROJECT_ROOT}/src/fine-tuning_(training)"

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "增强版训练：TokenMixUp + 多尺度 + 域降质"
echo "=========================================="
echo "项目目录: ${PROJECT_ROOT}"
echo "脚本目录: ${SCRIPT_DIR}"
echo "开始时间: $(date)"
echo ""

# 启动 TensorBoard
TB_LOG_DIR="${PROJECT_ROOT}/data/logs/tensorboard"
mkdir -p "${TB_LOG_DIR}"

TB_RUNNING=$(ps aux | grep "tensorboard.*6007" | grep -v grep)
if [ -z "${TB_RUNNING}" ]; then
    tensorboard --logdir "${TB_LOG_DIR}" --port 6007 --bind_all &
    echo "TensorBoard 已启动: http://localhost:6007"
else
    echo "TensorBoard 已在运行"
fi
echo ""

echo "启动增强训练..."
python train_lora_augmented.py

echo ""
echo "=========================================="
echo "增强训练完成!"
echo "结束时间: $(date)"
echo "=========================================="
