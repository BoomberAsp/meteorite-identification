import os


current_path = os.path.dirname(os.path.abspath(__file__))
# 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)

TEST_DATA_DIR = os.path.join(root_path, "data/test_images")
print(TEST_DATA_DIR)