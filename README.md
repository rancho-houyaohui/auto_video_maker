---
title: AI Video Workstation (AI 全栈视频工作站)
description: 专为制作高密度、强视觉冲击、快节奏短视频而生的全流程自动化工具
category: 视频制作工具
tags: [AI视频制作, 自动化视频, 本地大模型, Ollama, Python, FFmpeg]
author: ranchohou
date: 2026-01-18
version: 1.0.0
---

# 🎬 AI Video Workstation (AI 全栈视频工作站)

> 专为制作高密度、强视觉冲击、快节奏短视频而生的全流程自动化工具。

这是一个基于本地大模型 (Ollama) 和 Python 的可视化视频生产系统。它不仅能自动生成视频，更提供了一个**“所见即所得”**的 Web 控制台，允许你对每一个分镜、每一句台词、每一个音效进行精细化调整。

这里是这个视频制作工具的[视频功能讲解](https://www.bilibili.com/video/BV1rF6MBdEgc/)。

## ✨ 核心功能

*   **🧠 AI 深度拆解**：调用本地 Qwen2.5 模型，智能拆解文案，提取视觉关键词、重点金句，并自动规划音效。
*   **🎨 动态字幕系统**：
    *   **普通模式**：底部白字，支持自动识别文案中的关键词并高亮标红。
    *   **霸屏模式**：核心金句屏幕居中、大字号、红字白边、淡入淡出动效。
*   **📦 智能资源库**：
    *   **视频**：集成 Pexels API，支持在线搜索预览，点击即下载至本地库。支持“懒加载”渲染。
    *   **音效**：集成 MyInstants 搜索，无需 Key 即可搜索海量 Meme 音效。
    *   **音乐**：支持本地音乐上传、预览和管理，可作为背景音乐使用。
    *   **本地优先**：自动建立本地资产索引，越用越快。
*   **🎛️ 可视化精修**：
    *   提供 Web UI 界面，支持拖拽拆分分镜、实时预览字幕样式、试听音效。
    *   支持调节 TTS 语速、句间停顿（呼吸感）、背景音乐音量。
*   **🚀 实时渲染反馈**：通过 WebSocket 实时展示 FFmpeg 渲染进度，拒绝黑盒等待。
*   **💾 项目管理**：自动保存项目历史，支持回溯编辑、重新渲染、一键删除。
*   **🎨 封面制作**：支持上传背景图、添加主标题和副标题、调整字体样式和颜色、导出高清封面图。
*   **⚡ 批量生成**：支持自动处理状态为'draft'的项目，执行 Analyze -> Render 流程。
*   **📊 历史记录**：支持查看和管理生成的视频历史，包括视频详情、大小、生成时间等信息。

---

## 🛠️ 安装与部署

### 1. 环境要求
*   **系统**：macOS (推荐 M1/M2/M3/M4 芯片) / Linux / Windows
*   **Python**：3.10 或更高版本
*   **Conda**：推荐使用 Conda 管理 Python 环境。
*   **基础软件**：
    *   [FFmpeg](https://ffmpeg.org/) (视频处理核心)
    *   [ImageMagick](https://imagemagick.org/) (字幕渲染核心)
    *   [Ollama](https://ollama.com/) (本地 LLM 运行环境)

### 2. 安装步骤
#### 2.1 环境准备 (推荐 Conda)

强烈建议使用 Conda 来管理 Python 环境，以避免包依赖冲突。

##### 2.1.1 安装 Conda (如果尚未安装)
请访问 [Conda 官网](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html) 下载并安装。

##### 2.1.2 创建并激活虚拟环境
在项目根目录 (`auto_video_maker`) 下操作：

```bash
# 1. 创建新环境 (例如，命名为 cosyvoice)，指定 Python 版本
conda create -n cosyvoice python=3.10 -y

# 2. 激活环境 (每次在新终端启动项目前都需要执行)
conda activate cosyvoice
```

#### 2.2 Mac 用户一键安装 (推荐)
```bash
# 1. 安装系统依赖
brew install ffmpeg imagemagick

# 2. 安装 Python 依赖
pip install -r requirements.txt
# (如果 requirements.txt 缺失，请手动安装：pip install fastapi uvicorn python-multipart jinja2 aiofiles requests ollama edge-tts moviepy==1.0.3 numpy pillow websockets)

# 3. 准备字体 (关键!)
# 请下载一个中文字体(如阿里普惠体-Heavy.ttf)，重命名为 font.ttf
# 放入 assets/fonts/ 目录下
mkdir -p assets/fonts
# cp /path/to/your/font.ttf assets/fonts/font.ttf
```

### 3. 模型准备
本系统依赖 `qwen2.5:7b` 模型进行文本分析。请确保 Ollama 已运行：
```bash
# 在终端执行
ollama pull qwen2.5:7b

# 安装 OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

python -c "import clip; print('CLIP 安装成功')"
```

---

## ⚙️ 配置指南

项目根目录下的 `config.py` 是核心配置文件。

1.  **API 密钥配置**：
    为了使用在线视频搜索，你需要申请 Pexels API Key（免费）。
    打开 `config.py` 修改：
    ```python
    PEXELS_API_KEY = "你的_Pexels_API_Key"
    # PIXABAY_API_KEY = "..." (音效已切换为免Key源，此处可留空)
    ```

2.  **参数微调** (可选)：
    ```python
    DEFAULT_VOICE = "zh-CN-YunxiNeural" # 默认配音角色
    BGM_VOLUME = 0.1                    # 背景音乐默认音量
    DEFAULT_TTS_RATE = "+10%"           # 默认语速
    ```

---

## ▶️ 启动与使用

### 1. 启动服务
在项目根目录下运行：
```bash
python server.py
```
终端显示 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。

### 2. 页面导航
打开浏览器访问 **`http://127.0.0.1:8000`**，系统提供以下核心页面：

*   **🏠 首页**：系统概览，快速访问各功能模块。
*   **📝 分镜精修**：AI 分析文案、拆解分镜、精修调整的核心页面。
*   **🎨 封面制作**：专门用于创建视频封面的可视化编辑器。
*   **📦 资源库**：管理本地和在线视频、音效、音乐资源。
*   **💾 项目管理**：查看、编辑、删除项目，管理项目状态。
*   **📜 历史记录**：查看和管理生成的视频历史。
*   **❓ 帮助中心**：系统使用指南和常见问题解答。

### 3. 操作流程

#### 3.1 全剧配置
*   在 **“⚙️ 配置”** 面板中，确认 API Key 是否生效。
*   上传或选择背景音乐 (BGM)，设置全局音量和语速。

#### 3.2 AI 分析与分镜精修
*   进入 **“📝 分镜精修”** 面板。
*   输入你的文案（支持长文本）。
*   点击 **“🧠 AI 拆解分镜”**。系统会自动提取关键词、匹配画面和音效。
*   在下方卡片中：
        *   修改文案、关键词。
        *   拖动文字调整位置。
        *   更换视频素材（本地或在线 Pexels 搜索下载）。
        *   选择音效。
        *   配置每个分镜的停顿和语速。
        *   设置“重点模式”开关。

#### 3.3 精修与调整 (核心)
*   **检查画面**：点击视频缩略图，如果不满意，可以在弹窗中搜索 Pexels 在线素材并下载替换。
*   **调整字幕**：
    *   **普通句**：确保关键词提取正确，预览框会显示底部白字+红字高亮。
    *   **重点句**：勾选 **“重点霸屏模式”**，预览框会显示屏幕居中大字。
*   **调整节奏**：觉得某句话太快？单独调整该分镜的“停顿”时间。
*   **试听音效**：点击音效下拉框，选择合适的转场声（如 Whoosh, Boom）。

#### 3.4 封面制作
*   进入 **“🎨 封面制作”** 面板。
*   上传背景图片或使用演示图片。
*   添加主标题和副标题，调整字体大小、颜色、描边等样式。
*   拖动文字调整位置（支持中心吸附功能）。
*   选择导出倍率（1x、1.5x、2x），点击 **“导出封面图”** 保存成品。

#### 3.5 资源管理
*   进入 **“📦 资源库”** 面板。
*   **视频资源**：浏览本地视频库，或通过 Pexels 在线搜索并下载视频素材。
*   **音效资源**：浏览本地音效库，或通过 MyInstants 在线搜索并下载音效。
*   **音乐资源**：上传本地音乐文件，预览和管理背景音乐。

#### 3.6 渲染与导出
*   在 **“📝 分镜精修”** 面板中，点击右上角的 **“🚀 开始渲染”**。
*   观察黑色弹窗中的实时进度日志。
*   渲染完成后，点击 **“📥 下载视频”** 保存成品。
*   生成的文件会自动归档在 **“📜 历史记录”** 面板，支持随时回看或重新编辑。

#### 3.7 批量生成
*   确保项目状态设置为 **“draft”**。
*   系统会自动处理状态为 **“draft”** 的项目，执行 Analyze -> Render 流程。
*   生成的视频会存储到项目 **“publish_time”** 指定的文件夹中。

---

## 📦 打包与分发

### 1. 安装打包工具
```bash
# 安装 PyInstaller
pip install pyinstaller
```

### 2. 准备打包资源
确保以下资源已正确放置：
- `assets/fonts/font.ttf` - 中文字体文件
- `assets/` 目录下的所有必要资源
- `templates/` 目录下的前端页面
- `bin/` 目录下的 FFmpeg 可执行文件（根据系统类型）

### 3. 执行打包命令
```bash
# 基本打包命令
pyinstaller --name "AI_Video_Workstation" \
    --add-data "assets:assets" \
    --add-data "templates:templates" \
    --add-data "bin:bin" \
    --onefile \
    --windowed \
    server.py

# 确保 FFmpeg 可执行权限
chmod +x bin/ffmpeg
# 清理旧打包文件
rm -rf build dist
# 重新打包
pyinstaller AI_Video_Station.spec --clean --noconfirm

# 检查打包文件
./dist/AI_Video_Station.app/Contents/MacOS/AI_Video_Station
```

**参数说明：**
- `--name` - 打包后的应用名称
- `--add-data` - 添加需要包含的资源目录
- `--onefile` - 生成单个可执行文件
- `--windowed` - 不显示命令行窗口（仅 GUI 应用）

### 4. 运行打包后的应用
打包完成后，可执行文件将生成在 `dist/` 目录下：
```bash
# macOS
open dist/AI_Video_Workstation.app

# Windows
dist/AI_Video_Workstation.exe

# Linux
dist/AI_Video_Workstation
```

### 5. 打包注意事项
- 确保 FFmpeg 已正确放入 `bin/` 目录，并且具有执行权限
- 打包过程中可能需要较长时间，请耐心等待
- 首次运行打包后的应用可能需要初始化资源，速度较慢

---

## � 目录结构说明

```text
auto_video_maker/
├── __pycache__/         # Python 编译缓存
├── assets/              # 资源文件夹
│   ├── video/           # 本地视频素材库 (自动下载/手动放入)
│   ├── music/           # 背景音乐库 (支持网页上传)
│   ├── sfx/             # 音效库 (自动下载/手动放入)
│   ├── fonts/           # 字体库 (必须包含 font.ttf)
│   └── exports/         # (预留) 导出缓存
├── bin/                 # 二进制文件目录 (包含 ffmpeg 等)
├── build/               # 构建输出目录
├── build_assets/        # 构建资源目录
│   ├── video/           # 构建用视频素材
│   ├── music/           # 构建用音乐素材
│   ├── sfx/             # 构建用音效素材
│   ├── fonts/           # 构建用字体
│   └── icon.icns        # 应用图标
├── outputs/             # 最终生成的 MP4 视频及 JSON 项目文件
├── temp_web/            # 临时渲染缓存 (每次渲染会自动清理)
├── templates/           # 前端 HTML 页面
├── AI_Video_Station.spec # PyInstaller 打包配置
├── config.py            # 全局配置文件
├── desktop_app.py       # 桌面应用入口 (WebView 启动)
├── entitlements.plist   # macOS 权限配置
├── install.sh           # 安装脚本
├── logic.py             # 核心视频处理逻辑 (Engine)
├── project_manager.py   # 项目管理模块 (加载/保存项目)
├── projects.json        # 项目配置文件 (存储用户分镜配置)
├── requirements.txt     # Python 依赖列表
├── server.py            # Web 服务器接口 (API)
└── vfx_core.py          # 视觉特效处理核心模块
```


## ❓ 常见问题排查

**Q: 报错 `Error opening font: ... PingFangUI.ttc`?**
A: 这是因为 FFmpeg 无法读取 macOS 系统字体。
**解决**：必须手动下载一个 `.ttf` 字体文件（推荐造字工房劲黑或阿里普惠体粗体），重命名为 `font.ttf`，放入 `assets/fonts/` 文件夹中。

**Q: 点击“在线搜索”没反应？**
A: 1. 检查 `config.py` 中是否填入了有效的 `PEXELS_API_KEY`。
2. 检查网络是否能访问 `api.pexels.com`。

**Q: 渲染到一半报错 `Connection timeout`?**
A: 通常是 Edge-TTS 连接微软服务器超时。程序内置了重试机制，重新点击渲染通常即可解决。

**Q: 视频画面黑屏？**
A: 检查该分镜是否分配了有效的视频素材。如果本地文件被误删，请重新选择素材。



### 如何更新到您的项目

1.  在根目录新建 `requirements.txt`，粘贴上述内容。
2.  在根目录新建 `README.md`，粘贴上述内容。
3.  确保 `assets/fonts/` 目录下有一个 `font.ttf` 文件（如果没有，程序会降级使用 Arial，但在中文环境下 Arial 可能会显示乱码或很丑，所以**务必放一个中文字体**）。
