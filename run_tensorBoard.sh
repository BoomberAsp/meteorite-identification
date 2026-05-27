#!/bin/bash
PROJECT_ROOT="/root/Projects/meteorite-identification"
SCRIPT_DIR="${PROJECT_ROOT}/src/fine-tuning_(training)"
cd "${SCRIPT_DIR}"
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