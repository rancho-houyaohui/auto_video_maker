import os
import sys
import platform
import stat
import shutil

# --- 路径寻址函数 ---
def get_resource_path(relative_path):
    """获取资源绝对路径 (兼容 PyInstaller 打包环境)"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的临时解压目录
        base_path = sys._MEIPASS
    else:
        # 本地开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- 环境配置 ---
system_name = platform.system() # 'Darwin' (Mac) or 'Windows'
IS_FROZEN = getattr(sys, 'frozen', False) # 是否为打包环境

# 环境类型：development (开发环境) 或 production (生产环境)
# 开发环境：IS_FROZEN = False
# 生产环境：IS_FROZEN = True
ENVIRONMENT = 'production' if IS_FROZEN else 'development'

FFMPEG_BINARY = None

if IS_FROZEN:
    # A. 打包模式：使用包内的 bin 目录
    ffmpeg_filename = "ffmpeg.exe" if system_name == 'Windows' else "ffmpeg"
    bundled_path = get_resource_path(os.path.join("bin", ffmpeg_filename))
    
    if os.path.exists(bundled_path):
        FFMPEG_BINARY = bundled_path
        
        # 自动修复权限 (仅限 Mac/Linux 打包环境)
        if system_name != 'Windows':
            try:
                st = os.stat(FFMPEG_BINARY)
                if not (st.st_mode & stat.S_IEXEC):
                    print(f"🔧 Fixing bundled FFmpeg permissions...")
                    os.chmod(FFMPEG_BINARY, st.st_mode | stat.S_IEXEC)
            except: pass
    else:
        print(f"⚠️ Warning: Bundled FFmpeg not found at {bundled_path}")

else:
    # B. 本地开发模式：查找系统环境变量中的 ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        FFMPEG_BINARY = system_ffmpeg
        print(f"🔧 Using System FFmpeg: {system_ffmpeg}")
    else:
        # 如果系统没装，回退尝试找 bin 目录 (开发时也可以手动放 bin)
        local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "ffmpeg")
        if os.path.exists(local_bin):
            FFMPEG_BINARY = local_bin
        else:
            print("❌ Error: FFmpeg not found in PATH or local bin folder.")

# --- 用户数据目录 (生成的视频存这里) ---
# Windows: C:\Users\Name\Documents\AI_Video_Output
# Mac: /Users/Name/Documents/AI_Video_Output
USER_DOCS = os.path.join(os.path.expanduser("~"), "Documents", "AI_Video_Output")

# --- 目录配置 ---
ASSETS_DIR = get_resource_path("assets")
TEMPLATE_DIR = get_resource_path("templates")

# 动态产物目录
if IS_FROZEN:
    # 打包模式：使用用户文档目录
    OUTPUT_DIR = os.path.join(USER_DOCS, "outputs")
    TEMP_DIR = os.path.join(USER_DOCS, "temp_web")
    PROJECT_DB_FILE = os.path.join(USER_DOCS, "projects.json") 
else:
    # 本地开发模式：使用项目根目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(base_dir, "outputs")
    TEMP_DIR = os.path.join(base_dir, "temp_web")
    PROJECT_DB_FILE = os.path.join(base_dir, "projects.json")

# 确保目录存在 - 合并所有需要创建的目录，避免重复创建
all_dirs = [USER_DOCS, OUTPUT_DIR, TEMP_DIR]
for d in all_dirs:
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except PermissionError:
            print(f"⚠️  权限不足，无法创建目录: {d}")
            # 不要抛出异常，让应用继续运行，后续操作可能会再次尝试或使用其他目录
        except Exception as e:
            print(f"❌ 创建目录失败: {d}, 错误: {e}")

# 字体路径
FONT_PATH = os.path.join(ASSETS_DIR, "fonts", "font.ttf")



# --- API Keys (在此填入，前端如果没有输入则默认使用这里的) ---
PEXELS_API_KEY = "" 
PIXABAY_API_KEY = "" # 用于下载音效

# --- LLM 配置 ---
# MODEL_NAME = "qwen2.5:7b"  # 本地 Ollama 模型名称

# 模式选择: "ollama" (本地) 或 "api" (云端)
LLM_PROVIDER = "ollama" 

# 1. Ollama 配置
OLLAMA_MODEL = "qwen2.5:7b"

# 2. API 配置 (兼容 OpenAI 格式，支持 DeepSeek, ChatGPT, SiliconFlow 等)
API_BASE_URL = "https://api.deepseek.com/v1"  # 例如 DeepSeek 地址
API_KEY = ""                               # 在此填入 API Key
API_MODEL_NAME = "deepseek-chat"           # 模型名称

# --- 默认音频设置 ---
DEFAULT_VOICE = "zh-CN-YunxiNeural"  # 默认解说声音
BGM_VOLUME = 0.06                     # 背景音乐音量
AUDIO_PADDING = 0                  # 句间停顿(秒)
DEFAULT_TTS_RATE = "+40%"            # 默认语速

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
