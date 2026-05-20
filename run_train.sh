#!/bin/bash
# run_train.sh - 启动陨石识别模型 LoRA 微调训练

# 项目根目录
PROJECT_ROOT="/root/Projects/meteorite-identification"
SCRIPT_DIR="${PROJECT_ROOT}/src/fine-tuning_(training)"

# 设置 CUDA 可见设备（可选，根据实际情况调整）
# export CUDA_VISIBLE_DEVICES=0,1

# 切换到脚本目录
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "陨石识别模型 LoRA 微调训练"
echo "=========================================="
echo "项目目录: ${PROJECT_ROOT}"
echo "脚本目录: ${SCRIPT_DIR}"
echo "开始时间: $(date)"
echo ""

# 启动 TensorBoard（端口 6007）
echo "启动 TensorBoard (端口 6007)..."
TB_LOG_DIR="${PROJECT_ROOT}/data/logs/tensorboard"
mkdir -p "${TB_LOG_DIR}"

# 检查 TensorBoard 是否已在 6007 端口运行
TB_RUNNING=$(ps aux | grep "tensorboard.*6007" | grep -v grep)
if [ -z "${TB_RUNNING}" ]; then
    tensorboard --logdir "${TB_LOG_DIR}" --port 6007 --bind_all &
    echo "TensorBoard 已启动: http://localhost:6007"
else
    echo "TensorBoard 已在运行"
fi
echo ""

# 启动训练
echo "启动训练..."
python train_lora.py

echo ""
echo "=========================================="
echo "训练完成!"
echo "结束时间: $(date)"
echo "=========================================="