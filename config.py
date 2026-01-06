# config.py

# --- API Keys (在此填入，前端如果没有输入则默认使用这里的) ---
PEXELS_API_KEY = "" 
PIXABAY_API_KEY = "" # 用于下载音效

# --- LLM 配置 ---
MODEL_NAME = "qwen2.5:7b"  # 本地 Ollama 模型名称

# --- 默认音频设置 ---
DEFAULT_VOICE = "zh-CN-YunxiNeural"  # 默认解说声音
BGM_VOLUME = 0.1                     # 背景音乐音量
AUDIO_PADDING = 0                  # 句间停顿(秒)
DEFAULT_TTS_RATE = "+10%"            # 默认语速

# --- 可选语音列表 (用于前端下拉菜单) ---
# 格式: ("标识符", "显示名称")
VOICE_OPTIONS = [
    ("zh-CN-YunxiNeural", "云希 - 沉稳男声"),
    ("zh-CN-XiaoxiaoNeural", "晓晓 - 活泼女声"),
    ("zh-CN-YunjianNeural", "云健 - 体育/激昂男声"),
    ("zh-CN-YunyangNeural", "云扬 - 新闻播音男"),
    ("zh-CN-Liaoning-XiaobeiNeural", "小北 - 东北话(趣味)"),
    ("zh-TW-HsiaoChenNeural", "晓臻 - 台湾女声"),
]

# --- 路径配置 ---
ASSETS_DIR = "./assets"
TEMP_DIR = "temp_web"
FONT_PATH = "./assets/fonts/font.ttf"

# 合成视频存放目录
OUTPUT_DIR = "outputs"
