#!/bin/bash
# run_train_baseline.sh - 基线训练：无增强方法对照实验

PROJECT_ROOT="/root/Projects/meteorite-identification"
SCRIPT_DIR="${PROJECT_ROOT}/src/fine-tuning_(training)"

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "基线训练：无增强方法对照"
echo "  无 TokenMixUp / 无多尺度 / 无域降质"
echo "=========================================="
echo "项目目录: ${PROJECT_ROOT}"
echo "脚本目录: ${SCRIPT_DIR}"
echo "开始时间: $(date)"
echo ""

# 启动 TensorBoard（端口 6007）
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

echo "启动基线训练..."
python train_lora_baseline.py

echo ""
echo "=========================================="
echo "基线训练完成!"
echo "结束时间: $(date)"
echo "=========================================="
