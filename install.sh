#!/bin/bash

echo "🚀 开始部署 AI 视频工作站..."

# 1. 检查 Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ 未检测到 Homebrew，请先安装 Homebrew: https://brew.sh/"
    exit 1
fi

# 2. 安装系统级依赖 (FFmpeg, ImageMagick)
echo "📦 正在安装系统依赖 (FFmpeg, ImageMagick)..."
# 设置 Homebrew 国内镜像源 (可选，防止卡住)
export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.tuna.tsinghua.edu.cn/homebrew-bottles
brew install ffmpeg imagemagick

# 3. 创建目录结构
echo "dv 正在初始化目录结构..."
mkdir -p assets/video
mkdir -p assets/music
mkdir -p assets/sfx
mkdir -p assets/fonts
mkdir -p outputs
mkdir -p temp_web
mkdir -p temp_scenes

# 4. 自动下载中文字体 (防止字幕报错)
FONT_FILE="assets/fonts/font.ttf"
if [ ! -f "$FONT_FILE" ]; then
    echo "⬇️ 未检测到字体，正在下载免费商用字体 (阿里普惠体)..."
    # 这里使用一个稳定的 GitHub 镜像源或 CDN
    curl -L "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Simplified/NotoSansSC-Bold.otf" -o "$FONT_FILE"
    # 或者如果下载失败，提示用户
    if [ $? -ne 0 ]; then
        echo "⚠️ 字体下载失败，请手动下载一个 .ttf 中文字体，重命名为 font.ttf 放入 assets/fonts/ 目录。"
    else
        echo "✅ 字体下载完成！"
    fi
else
    echo "✅ 字体文件已存在。"
fi

# 5. 安装 Python 依赖
echo "🐍 正在安装 Python 依赖 (使用清华源)..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "========================================"
echo "🎉 安装完成！"
echo "请确保您已安装并运行了 Ollama (qwen2.5:7b)"
echo "运行方式: python server.py"
echo "========================================"