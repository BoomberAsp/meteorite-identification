from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from modelscope import snapshot_download

# 下载Qwen3-VL-8B-Instruct模型到models目录
model_path = snapshot_download('qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
print(f"Model downloaded to: {model_path}")
