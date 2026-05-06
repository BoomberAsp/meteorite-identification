"""
ploting.py

绘制训练过程中的 loss 曲线：
- Training loss 曲线
- Validation loss 曲线
"""

import os
import re
import ast
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def parse_loss_file(loss_file_path: str):
    """
    解析 loss.txt 文件，提取训练和验证 loss

    Args:
        loss_file_path: loss.txt 文件路径

    Returns:
        train_data: [(epoch, loss), ...]
        eval_data: [(epoch, eval_loss), ...]
    """
    train_data = []
    eval_data = []

    with open(loss_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 格式: {dict-like string}
            # 例如: {'loss': 22.783, 'grad_norm': 42.64, 'learning_rate': 1.75e-06, 'epoch': 0.01}
            line = line.strip()
            if not line:
                continue

            # 解析 dict-like 字符串
            try:
                # 使用 ast.literal_eval 安全解析
                data_dict = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue

            epoch = data_dict.get('epoch', None)

            # 训练 loss
            if 'loss' in data_dict:
                train_data.append((epoch, data_dict['loss']))

            # 验证 loss
            if 'eval_loss' in data_dict:
                eval_data.append((epoch, data_dict['eval_loss']))

    return train_data, eval_data


def plot_loss_curves(train_data, eval_data, output_path: str):
    """
    绘制 loss 曲线并保存

    Args:
        train_data: [(epoch, loss), ...]
        eval_data: [(epoch, eval_loss), ...]
        output_path: 输出图片路径
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 提取数据
    train_epochs = [d[0] for d in train_data]
    train_losses = [d[1] for d in train_data]

    eval_epochs = [d[0] for d in eval_data]
    eval_losses = [d[1] for d in eval_data]

    # 绘制训练 loss
    ax.plot(train_epochs, train_losses, 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')

    # 绘制验证 loss（使用更大的点标记）
    ax.plot(eval_epochs, eval_losses, 'r-', linewidth=2, markersize=8, label='Validation Loss')
    ax.scatter(eval_epochs, eval_losses, color='red', s=50, zorder=5)

    # 设置标题和标签
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置图例
    ax.legend(loc='upper right', fontsize=11)

    # 设置 y 轴范围（可选，让曲线更清晰）
    min_loss = min(min(train_losses), min(eval_losses)) - 0.5
    max_loss = max(train_losses[:10]) + 1  # 使用前10个训练 loss 的最大值，避免初始高值影响
    ax.set_ylim(min_loss, max_loss)

    # 添加统计信息文本
    initial_train_loss = train_losses[0]
    final_train_loss = train_losses[-1]
    final_eval_loss = eval_losses[-1]

    stats_text = f"Initial Training Loss: {initial_train_loss:.2f}\n"
    stats_text += f"Final Training Loss: {final_train_loss:.2f}\n"
    stats_text += f"Final Validation Loss: {final_eval_loss:.2f}\n"
    stats_text += f"Training Steps: {len(train_data)}\n"
    stats_text += f"Evaluation Steps: {len(eval_data)}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to: {output_path}")

    plt.close()


def main():
    # 路径配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    data_root = os.path.join(project_root, "data")

    loss_file = os.path.join(project_root, "loss.txt")
    output_dir = os.path.join(data_root, "logs")
    output_file = os.path.join(output_dir, "loss_curves.png")

    # 检查输入文件
    if not os.path.exists(loss_file):
        print(f"Error: loss.txt not found at {loss_file}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 解析数据
    print(f"Parsing loss file: {loss_file}")
    train_data, eval_data = parse_loss_file(loss_file)

    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")

    if len(train_data) == 0 and len(eval_data) == 0:
        print("Error: No data found in loss file")
        return

    # 绘制曲线
    plot_loss_curves(train_data, eval_data, output_file)


if __name__ == "__main__":
    main()