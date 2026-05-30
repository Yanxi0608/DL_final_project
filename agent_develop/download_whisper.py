# download_model.py
import os
import urllib.request
from pathlib import Path

MODEL_NAME = "ggml-base.bin"  # 可以改成 tiny, base, small 等
MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{MODEL_NAME}"
# 备用镜像源（如果国内访问 HuggingFace 慢）：
# MODEL_URL = f"https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/{MODEL_NAME}"

def download_whisper_model():
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    target_path = models_dir / MODEL_NAME
    if target_path.exists():
        print(f"✅ 模型 {MODEL_NAME} 已存在，无需下载。")
        return

    print(f"⏳ 正在下载 Whisper 模型到 {target_path}，请稍候...")
    try:
        urllib.request.urlretrieve(MODEL_URL, target_path)
        print("🎉 下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

def get_whisper_model_path(model_name: str = MODEL_NAME) -> Path:
    model_dir = Path("./models")
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录 {model_dir} 不存在，请先运行 download_whisper_model() 下载模型。")
    return model_dir / model_name



# if __name__ == "__main__":
#     download_whisper_model()