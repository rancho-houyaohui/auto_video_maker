# 🎬 AI Video Workstation (AI 视频全流程工作站)

这是一个基于本地大模型 (Ollama) 和 Python 的自动化视频剪辑工具。它能根据文案自动分析分镜、匹配素材、生成语音(TTS)、添加音效、背景音乐，并渲染出带特效字幕的短视频。

## ✨ 特性
- **全流程自动化**：文案 -> 视频，一键搞定。
- **可视化界面**：Web UI 支持分镜精修、实时预览、样式调整。
- **本地优先**：素材库自动积累，越用越快。
- **智能分析**：使用 Qwen2.5 进行分镜拆解和重点词提取。
- **Mac M4 优化**：针对 Apple Silicon 进行 FFmpeg 硬件加速配置。

## 🛠️ 环境准备

1. **系统**：macOS (推荐 M1/M2/M3/M4)
2. **Python**：3.10 或更高版本
3. **Ollama**：
   - 请先安装 [Ollama](https://ollama.com/)
   - 拉取模型：在终端运行 `ollama pull qwen2.5:7b`

## 🚀 快速安装

在项目根目录下打开终端，运行一键安装脚本：

```bash
chmod +x install.sh
./install.sh
```

该脚本会自动：
*   安装 `ffmpeg` 和 `imagemagick`。
*   安装所有 Python 依赖库。
*   下载默认中文字体。
*   初始化资源目录。

创建并激活 Conda 环境 
```bash
# 创建一个名为 cosyvoice 的环境，并安装 Python 3.9
conda create -n cosyvoice python=3.9
# 先激活环境
conda activate cosyvoice
```

## ⚙️ 配置 (Config)

打开 `config.py` 文件，填入您的 API Keys 以启用在线下载功能：

```python
# 必填：用于下载高清视频素材
PEXELS_API_KEY = "你的_key"

# 必填：用于下载音效
PIXABAY_API_KEY = "你的_key"
```

*   **Pexels Key 获取**：[https://www.pexels.com/api/](https://www.pexels.com/api/)
*   **Pixabay Key 获取**：[https://pixabay.com/api/docs/](https://pixabay.com/api/docs/)

## ▶️ 启动运行

1. 启动服务：
   ```bash
   python server.py
   ```

2. 访问界面：
   打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

3. 封面图绘制：
   打开浏览器访问：[http://127.0.0.1:8000/canvas.html](http://127.0.0.1:8000/canvas.html)

## 📖 使用流程

1. **输入文案**：在网页输入框粘贴你的视频脚本。
2. **AI 分析**：点击“AI 拆解分镜”，等待生成分镜列表。
3. **精修**：
   - **换视频**：点击视频缩略图，选择本地素材或在线下载。
   - **换声音**：选择不同的解说音色。
   - **调节奏**：拆分过长的句子，调整句间停顿。
   - **定样式**：在“字幕样式”Tab 调整字号和颜色。
4. **渲染**：点击“开始渲染”，查看实时终端日志，等待出片。

## 📂 目录说明

*   `assets/video`：存放所有下载的视频素材（文件名包含标签）。
*   `assets/music`：存放背景音乐（.mp3），可手动添加。
*   `assets/sfx`：存放音效文件。
*   `assets/fonts`：存放字幕字体（必须包含 font.ttf）。

## ❓ 常见问题

**Q: 报错 `Error opening font`?**
A: 请检查 `assets/fonts/` 目录下是否有 `font.ttf` 文件。如果没有，请手动下载一个中文字体放入。

**Q: 渲染时卡住不动？**
A: 请查看终端日志。如果是下载素材卡住，请检查网络是否能访问 Pexels。

**Q: 提示 `Ollama connection refused`?**
A: 请确保您本地已经运行了 Ollama 软件。
```

---

### ✅ 最终交付操作流程

现在，您只需要做这几步即可在任何一台新 Mac 上跑起来：

1.  把整个文件夹拷过去。
2.  打开终端进入文件夹。
3.  运行 `./install.sh`。
4.  修改 `config.py` 填 Key。
5.  运行 `python server.py`。

享受您的自动化开发之旅！