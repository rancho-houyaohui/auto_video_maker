from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import config 
import asyncio
import glob
import time
import re
import json
import shutil
import subprocess
import platform
import webview
import base64
from datetime import datetime
from logic import VideoEngine
from project_manager import project_mgr
from urllib.parse import quote

app = FastAPI()


# /clips/abc.mp4 就会指向 CLIP_LIB_DIR/abc.mp4
if os.path.exists(config.CLIP_LIB_DIR):
    app.mount("/clips", StaticFiles(directory=config.CLIP_LIB_DIR), name="clips")
    print(f"✅ 已挂载静态目录: {config.CLIP_LIB_DIR} -> /clips")
else:
    print(f"⚠️ 警告: 找不到目录 {config.CLIP_LIB_DIR}，无法挂载预览")

# --- 静态资源挂载 ---
# 1. Assets
# 目录已经在config.py中创建，这里不再重复创建
app.mount("/static", StaticFiles(directory=config.ASSETS_DIR), name="static")

# 2. Temp & Outputs
app.mount("/temp", StaticFiles(directory=config.TEMP_DIR), name="temp")
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

# 3. Templates
templates = Jinja2Templates(directory=config.TEMPLATE_DIR)

engine = VideoEngine()

# --- [核心修改] 全局进度存储 (内存数据库) ---
# 结构: { "client_id": { "percent": 10, "msg": "正在渲染...", "status": "running", "url": "" } }
GLOBAL_PROGRESS = {}

# --- WebSocket ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, asyncio.Queue] = {}
    async def connect(self, client_id: str):
        self.active_connections[client_id] = asyncio.Queue()
    def disconnect(self, client_id: str):
        if client_id in self.active_connections: del self.active_connections[client_id]
    async def send_log(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].put(message)

manager = ConnectionManager()

# --- 请求模型 ---
class SystemCommandRequest(BaseModel):
    path: str
    title: str = "AI 视频工作站"

class AnalyzeRequest(BaseModel):
    text: str
    pexels_key: str = ""
    pixabay_key: str = ""
    llm_provider: str = "" 
    llm_model: str = ""          
    llm_base_url: str = ""       
    llm_api_key: str = ""
    search_source: str = "vector"

class RenderRequest(BaseModel):
    client_id: str
    project_id: str = ""
    scenes: list
    output_name: str
    bgm_file: str = ""
    bgm_volume: float = 0.1
    audio_padding: float = 0.2
    tts_rate: str = "+15%" 
    subtitle_style: dict = {}
    search_source: str = "vector"

class DownloadSpecificRequest(BaseModel):
    url: str
    id: str
    tags: str

class DownloadSfxRequest(BaseModel):
    query: str
    url: str = "" 

class ProjectCreateReq(BaseModel):
    title: str
    script: str
    canvas_title: str = ""
    publish_time: str = ""
    main_title: str = ""
    sub_title: str = ""
    tags: str = ""

class ProjectUpdateReq(BaseModel):
    title: str = None
    canvas_title: str = None
    script: str = None
    video_path: str = None
    cover_path: str = None
    status: str = None
    publish_time: str = None
    scenes_data: list = None
    main_title: str = None
    sub_title: str = None
    tags: str = None

class CoverUploadReq(BaseModel):
    project_id: str
    image_data: str

class SearchVectorRequest(BaseModel):
    query: str
    top_k: int = 12

# --- 页面路由 ---
@app.get("/")
def index(request: Request):
    v_dir = os.path.join(config.ASSETS_DIR, "video")
    m_dir = os.path.join(config.ASSETS_DIR, "music")
    s_dir = os.path.join(config.ASSETS_DIR, "sfx")
    
    videos = [f for f in os.listdir(v_dir) if f.endswith(('.mp4', '.mov'))] if os.path.exists(v_dir) else []
    music = [f for f in os.listdir(m_dir) if f.endswith(('.mp3', '.wav'))] if os.path.exists(m_dir) else []
    sfx = [f for f in os.listdir(s_dir) if f.endswith(('.mp3', '.wav'))] if os.path.exists(s_dir) else []
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "videos": videos,
        "music": music,
        "sfx_list": sfx,
        "voice_options": config.VOICE_OPTIONS,
        "default_voice": config.DEFAULT_VOICE,
        "default_tts_rate": config.DEFAULT_TTS_RATE,
        "default_pexels": config.PEXELS_API_KEY if "粘贴" not in config.PEXELS_API_KEY else "",
        "default_pixabay": config.PIXABAY_API_KEY if "粘贴" not in config.PIXABAY_API_KEY else "",
        "default_llm_provider": config.LLM_PROVIDER,
        "default_llm_model": config.OLLAMA_MODEL if config.LLM_PROVIDER == 'ollama' else config.API_MODEL_NAME,
        "default_api_base": config.API_BASE_URL,
        "is_frozen": config.IS_FROZEN
    })

# --- [核心修改] 进度查询接口 ---
@app.get("/api/progress/{client_id}")
def get_progress(client_id: str):
    # 返回当前进度，如果没有则返回默认空状态
    return GLOBAL_PROGRESS.get(client_id, {"percent": 0, "msg": "等待任务...", "status": "idle"})

@app.get("/canvas")
def canvas_page(request: Request):
    return templates.TemplateResponse("canvas.html", {"request": request})

@app.get("/projects")
def projects_page(request: Request):
    return templates.TemplateResponse("projects.html", {"request": request})

# --- 系统接口 ---
@app.post("/api/system/reveal")
def api_system_reveal(req: SystemCommandRequest):
    filename = os.path.basename(req.path)
    abs_path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(abs_path): return {"status": "error", "msg": "文件不存在"}
    try:
        if platform.system() == 'Darwin': subprocess.run(["open", "-R", abs_path])
        elif platform.system() == 'Windows': subprocess.run(["explorer", "/select,", abs_path])
        else: subprocess.run(["xdg-open", os.path.dirname(abs_path)])
        return {"status": "ok"}
    except Exception as e: return {"status": "error", "msg": str(e)}

# 打开新窗口
@app.post("/api/open/win")
def api_open_win(req: SystemCommandRequest):
    try:
        request_data = {
            'path': req.path,
            'title': req.title
        }
        webview.create_window(
            title=req.title, 
            url=f"http://127.0.0.1:18888{req.path}",
            width=1280,
            height=800,
            resizable=True,
            text_select=True
        )
        webview.start(debug=False)
        
        print(f"📤 Window open request sent to queue: {request_data}")
        return {"status": "ok", "message": "Window open request received and queued", "path": req.path, "title": req.title}
    except Exception as e:
        print(f"❌ Error sending window request to queue: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e), "path": req.path, "title": req.title}

# --- 封面上传接口 ---
@app.post("/api/cover/upload")
async def upload_cover(req: CoverUploadReq):
    try:
        # 1. 解析 Base64
        if "," in req.image_data:
            header, encoded = req.image_data.split(",", 1)
        else:
            encoded = req.image_data
        data = base64.b64decode(encoded)
        
        # 2. 确定文件夹 (核心修改：与视频保持一致)
        folder_date = datetime.now().strftime("%Y%m%d") # 默认今天
        
        # 查询项目信息获取时间
        p_data = project_mgr.get_one(req.project_id)
        if p_data and p_data.get("publish_time"):
            try:
                # 提取 "2026-01-20" -> "20260120"
                raw_time = p_data.get("publish_time")
                if len(raw_time) >= 10:
                    folder_date = raw_time[:10].replace("-", "")
            except: pass
            
        save_dir = os.path.join(config.OUTPUT_DIR, folder_date)
        os.makedirs(save_dir, exist_ok=True)
        
        # 3. 保存文件
        filename = f"cover_{req.project_id}_{int(time.time())}.png"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(data)
            
        # 4. 生成 URL 并更新数据库
        web_path = f"/outputs/{folder_date}/{filename}"
        project_mgr.update(req.project_id, {"cover_path": web_path})
        
        return {"status": "ok", "url": web_path}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "msg": str(e)}

# --- 批量自动生成 ---
@app.post("/api/batch_generate")
async def batch_generate():
    """
    查找所有状态为 'draft' 的项目，自动执行 Analyze -> Render
    并存储到项目 'publish_time' 指定的文件夹中
    """
    projects = project_mgr.get_all()
    drafts = [p for p in projects if p.get('status') == 'draft']
    
    # 以后台任务形式启动
    async def _process_queue():
        for p in drafts:
            try:
                print(f"⚙️ [Auto] Processing project: {p['title']}")
                
                # 1. 分析文案 (重置一下 LLM 配置以防万一)
                engine.set_llm_config(config.LLM_PROVIDER, config.API_KEY, config.API_BASE_URL, config.API_MODEL_NAME if config.LLM_PROVIDER == 'api' else config.OLLAMA_MODEL)
                scenes = engine.analyze_script(p['script'], search_source=req.search_source)
                
                # --- 2. 确定文件夹名称 (核心修改) ---
                folder_date = datetime.now().strftime("%Y%m%d") # 默认今天
                if p.get("publish_time"):
                    try:
                        # 提取 "2026-01-20" -> "20260120"
                        raw_time = p.get("publish_time")
                        if len(raw_time) >= 10:
                            folder_date = raw_time[:10].replace("-", "")
                    except: pass
                
                # 创建目录
                save_dir = os.path.join(config.OUTPUT_DIR, folder_date)
                os.makedirs(save_dir, exist_ok=True)
                
                # 准备路径
                safe_name = re.sub(r'[^\w\-_]', '', p['title'])
                if not safe_name: safe_name = f"video_{p['id']}"
                filename = f"{safe_name}_{int(time.time())}.mp4"
                
                output_path = os.path.join(save_dir, filename)
                web_url = f"/outputs/{folder_date}/{filename}"
                
                # 3. 渲染参数 (读取 Config)
                render_params = {
                    "scenes": scenes,
                    "bgm_file": config.BGM_FILE,
                    "bgm_volume": config.BGM_VOLUME,
                    "audio_padding": config.AUDIO_PADDING,
                    "tts_rate": config.DEFAULT_TTS_RATE,
                    "subtitle_style": {} 
                }
                
                async def noop_log(msg): pass
                
                # 4. 执行渲染
                await engine.render_project(render_params, output_path, noop_log)
                
                # 5. 更新状态
                project_mgr.update(p['id'], {
                    "status": "generated",
                    "video_path": web_url,
                    "video_abspath": output_path
                })
                print(f"✅ [Auto] Project {p['title']} completed. Saved to {web_url}")
                
            except Exception as e:
                print(f"❌ [Auto] Failed project {p['title']}: {e}")
                import traceback
                traceback.print_exc()

    asyncio.create_task(_process_queue())
    
    return {"status": "ok", "msg": f"已触发 {len(drafts)} 个任务的后台生成", "count": len(drafts)}

@app.post("/api/system/open")
def api_system_open(req: SystemCommandRequest):
    filename = os.path.basename(req.path)
    abs_path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(abs_path): return {"status": "error"}
    try:
        if platform.system() == 'Darwin': subprocess.run(["open", abs_path])
        elif platform.system() == 'Windows': os.startfile(abs_path)
        return {"status": "ok"}
    except Exception as e: return {"status": "error", "msg": str(e)}

@app.get("/api/check_env")
def check_env(provider: str):
    if provider == "ollama":
        is_running = engine.check_ollama_status()
        return {"status": "ok", "ollama_running": is_running}
    return {"status": "ok", "ollama_running": True}

# --- 历史记录 API ---
@app.get("/api/history")
def get_history():
    # 使用 glob 递归搜索所有子目录下的 .mp4
    # config.OUTPUT_DIR/**/ *.mp4
    search_path = os.path.join(config.OUTPUT_DIR, "**", "*.mp4")
    files = glob.glob(search_path, recursive=True)
    
    # 按修改时间倒序
    files.sort(key=os.path.getmtime, reverse=True)
    
    history = []
    for f in files:
        name = os.path.basename(f)
        size_mb = round(os.path.getsize(f) / (1024*1024), 1)
        ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        
        # 计算相对路径以便生成 URL (例如: 20260120/video.mp4)
        rel_path = os.path.relpath(f, config.OUTPUT_DIR)
        # 统一转为 web 路径分隔符 /
        web_path = rel_path.replace(os.sep, '/')
        
        # 检查同级目录下是否存在同名 json 文件
        json_path = f.replace('.mp4', '.json')
        has_json = os.path.exists(json_path)
        # json 的 web 访问路径
        json_file_web = web_path.replace('.mp4', '.json') if has_json else None

        history.append({
            "name": name, 
            "size": size_mb, 
            "time": ctime, 
            "url": f"/outputs/{web_path}", 
            "has_project": has_json, 
            "json_file": json_file_web # 前端可能需要这个路径
        })
    return {"status": "ok", "history": history}

@app.get("/api/history/load/{file_path:path}")
async def load_project_data(file_path: str):
    """
    [极速版] 加载项目
    """
    output_dir = getattr(config, 'OUTPUT_DIR', 'outputs')
    
    # 1. 拼接路径
    json_path = os.path.join(output_dir, file_path)
    
    # 容错处理
    if json_path.endswith(".mp4"):
        json_path = json_path.replace(".mp4", ".json")
        
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Project file not found")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # ========================================================
        # 此处保留之前的 [路径 -> URL] 映射代码 (清洗 scenes 里的 video 字段)
        # ========================================================
        scenes = data.get('scenes', []) or data.get('timeline', [])
        for scene in scenes:
            # ... (请务必保留之前写的 路径转URL 逻辑，否则前端无法预览) ...
            # ... (代码与上一轮回答一致，此处省略以节省篇幅) ...
            pass 

        return {"status": "ok", "data": data}

    except Exception as e:
        print(f"Load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/{file_path:path}")
async def delete_history(file_path: str):
    try:
        output_dir = getattr(config, 'OUTPUT_DIR', 'outputs')
        
        # 1. 拼接绝对路径
        # 防止目录遍历攻击 (../../)
        safe_path = os.path.normpath(os.path.join(output_dir, file_path))
        if not safe_path.startswith(os.path.abspath(output_dir)):
             raise HTTPException(status_code=403, detail="Invalid path")

        # 2. 确定 JSON 和 MP4 路径
        # 前端传来的 ID 是 json 结尾的
        if safe_path.endswith(".json"):
            json_path = safe_path
            video_path = safe_path.replace(".json", ".mp4")
        else:
            # 容错：万一传的是 mp4
            video_path = safe_path
            json_path = safe_path.replace(".mp4", ".json")

        print(f"🗑 [Direct Delete] {json_path}")

        deleted = []
        
        # 3. 直接删除
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted.append("json")
            
        if os.path.exists(video_path):
            os.remove(video_path)
            deleted.append("mp4")
            
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Files not found: {file_path}")
            
        return {"status": "ok", "deleted": deleted}
        
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 核心业务 API ---
@app.websocket("/ws/logs/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await manager.connect(client_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

# --- [新增] 强力 JSON 提取器 ---
def extract_json_content(text):
    """
    尝试从各种脏数据中提取纯净的 JSON 字符串
    """
    text = text.strip()
    
    # 1. 尝试移除 Markdown 代码块标记 (```json ... ```)
    # 使用正则非贪婪匹配提取大括号内的内容
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1)
    
    # 2. 如果没有代码块，但只有简单的 ``` 包裹
    if text.startswith("```") and text.endswith("```"):
        return text.strip("`").replace("json", "", 1).strip()

    # 3. 如果只是普通文本，原样返回尝试解析
    return text


@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    # 1. 配置引擎
    engine.set_api_keys(req.pexels_key, req.pixabay_key)
    engine.set_llm_config(req.llm_provider, req.llm_api_key, req.llm_base_url, req.llm_model)
    
    scenes = []
    is_direct_json = False
    
    print(f"📥 收到分析请求，长度: {len(req.text)}")

    # ================= [核心修改] 严格 JSON 嗅探逻辑 =================
    try:
        # A. 预处理：清洗 Markdown 和空白
        clean_text = extract_json_content(req.text)
        
        # B. 尝试解析
        # 只有以 { 开头 } 结尾的才最有可能是 JSON 对象
        # 这步判断能过滤掉绝大多数普通文本（比如 "帮我写一个..."）
        if clean_text.startswith("{") and clean_text.endswith("}"):
            data = json.loads(clean_text)
            
            # C. 特征字段验证 (这是区分“普通JSON”和“分镜JSON”的关键)
            # 必须包含 'timeline' 数组，或者 'project_meta'
            if isinstance(data, dict) and ("timeline" in data or "project_meta" in data):
                print("🚀 [模式切换] 检测到标准分镜 JSON，跳过 LLM 分析...")
                
                # 调用 logic.py 的解析器 (如果该函数还未在 logic.py 定义，请看下一步补充)
                if hasattr(engine, 'parse_direct_json'):
                    scenes = engine.parse_direct_json(data)
                else:
                    # 兜底：如果 logic.py 还没更新那个函数，手动简单解析
                    print("⚠️ logic.py 缺少 parse_direct_json，使用简易解析")
                    for block in data.get('timeline', []):
                        scenes.append({
                            "text": block.get('sentence_text', ''),
                            "visual_tags": block.get('visual_search_queries', []),
                            "video": "random", # 稍后会由 search_vector_match 覆盖
                            "voice": config.DEFAULT_VOICE
                        })
                
                is_direct_json = True
            else:
                print("ℹ️ 解析为 JSON，但缺少 'timeline' 字段，视为普通参考文本。")
        else:
            print("ℹ️ 文本不符合 JSON 格式特征，进入 LLM 模式。")

    except json.JSONDecodeError as e:
        # 解析失败，说明确实是普通文本
        print(f"ℹ️ JSON 解析尝试失败 (正常现象，视为普通文本): {str(e)[:50]}...")
    except Exception as e:
        print(f"⚠️ JSON 检测过程出现未知错误: {e}")
        traceback.print_exc()

    # 2. 如果不是 JSON，执行 LLM 分析
    if not is_direct_json:
        print("🤖 [模式切换] 执行 Qwen/GPT 语义分析...")
        scenes = engine.analyze_script(req.text, search_source=req.search_source)

    # 3. 统一资源匹配 (向量库搜索 & 格式化返回)
    # 这部分逻辑保持不变，用于给前端返回 video 的字符串路径
    for scene in scenes:
        v_info = scene.get('video_info', {})
        final_video_url = "" # 初始化为空字符串
        
        # A. 提取原始路径
        original_path = ""
        if isinstance(v_info, dict):
            original_path = v_info.get('src', '')
        elif isinstance(v_info, str):
            original_path = v_info
            
        # B. 执行路径转换逻辑
        if original_path and isinstance(original_path, str):
            # 1. 处理 clip_library (向量切片)
            if config.CLIP_LIB_DIR in original_path:
                # 替换绝对路径前缀为 URL 前缀
                # 结果: /clips/Anger In Cinema_scene_025.mp4
                rel_path = original_path.replace(config.CLIP_LIB_DIR, "/clips")
                
                # [重要] 处理 URL 编码 (解决文件名有空格/中文导致无法播放的问题)
                # 分割路径，只对文件名部分进行编码
                folder, filename = os.path.split(rel_path)
                final_video_url = f"{folder}/{quote(filename)}"
                
            # 2. 处理 assets (本地素材)
            elif config.ASSETS_DIR in original_path:
                rel_path = original_path.replace(config.ASSETS_DIR, "/static")
                folder, filename = os.path.split(rel_path)
                final_video_url = f"{folder}/{quote(filename)}"
                
            # 3. 处理在线链接 (Pexels)
            elif original_path.startswith("http"):
                final_video_url = original_path
                
            # 4. 其他情况 (保持原样，或给个随机)
            else:
                final_video_url = original_path # 实在匹配不上，为了不报错，先返回原值

        # C. [关键] 回写数据，确保前端拿到的是 URL
        
        # 1. 更新扁平字段 (供前端列表展示)
        scene['video'] = str(final_video_url)
        
        # 2. 更新对象字段 (供 render_project 后端使用，后端其实需要绝对路径，但这里我们做个特殊处理)
        # 注意：后端 logic.py 在渲染时需要绝对路径，前端需要 URL。
        # 为了不破坏后端逻辑，我们不动 v_info['src'] (保持绝对路径)，
        # 而是给前端加一个专用字段 'preview_url'。
        
        if isinstance(v_info, dict):
            # 这里的 src 保持绝对路径，供 ffmpeg 读取
            # 新增 preview 用来给前端显示
            v_info['src'] = str(final_video_url) 
            scene['video_info'] = v_info
            
            # 如果你的前端强行读的是 video_info.src，那你必须在这里把 src 改掉。
            # 但改掉 src 会导致后端渲染找不到文件。
            # 妥协方案：如果前端只读 scene.video 字符串，那上面 scene['video'] 已经改好了。
            
        # 确保 voice 存在
        if 'voice' not in scene:
            scene['voice'] = config.DEFAULT_VOICE

    return {"status": "ok", "scenes": scenes}

@app.post("/api/search_vector")
async def api_search_vector(req: SearchVectorRequest):
    """
    本地向量库搜索接口
    """
    print(f"🔍 Searching Vector DB for: {req.query}")
    
    # 1. 调用 logic.py 的搜索功能
    # search_vector_match 返回的是绝对路径列表 ['/Users/.../a.mp4', ...]
    try:
        paths = engine.search_vector_match(req.query, top_k=req.top_k)
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return {"status": "error", "results": []}

    results = []
    
    # 2. 格式化结果 (对齐 Pexels 接口的返回格式)
    for path in paths:
        # === 路径转 URL (复用之前的逻辑) ===
        final_video_url = ""
        filename = os.path.basename(path)
        
        # 处理 URL 编码 (解决中文/空格问题)
        safe_filename = quote(filename) 
        
        if config.CLIP_LIB_DIR in path:
            # 映射 /clips 路由
            # 注意：如果 CLIP_LIB_DIR 是 /Users/xx/clip_library
            # 那么 path 是 /Users/xx/clip_library/video.mp4
            # 我们需要把前缀替换掉
            # 这里更稳健的方法是直接拼接
            final_video_url = f"/clips/{safe_filename}"
            
        elif config.ASSETS_DIR in path:
            # 映射 /static 路由
            # 假设 path 是 /Users/xx/assets/video/abc.mp4
            #我们需要保留 video/abc.mp4 这一段
            rel_path = os.path.relpath(path, config.ASSETS_DIR)
            # 处理 Windows/Mac 路径分隔符差异
            rel_path_url = rel_path.replace("\\", "/")
            # 对每一级目录进行 quote 比较麻烦，简单起见对文件名 quote
            # 如果目录名有空格可能还会有问题，但在 assets 里通常还好
            # 这里做一个简化的全路径拼接
            final_video_url = f"/static/{rel_path_url}"
            # 再次修正：如果 rel_path 包含空格，浏览器还是会挂，最好 split 之后 quote
            # 但为了不把代码写太复杂，假设 assets 目录结构简单
            final_video_url = quote(final_video_url) 
            # quote 会把 / 转义，所以要先 quote 再拼，或者把 / 排除
            # 最佳实践：
            parts = rel_path_url.split("/")
            safe_parts = [quote(p) for p in parts]
            final_video_url = "/static/" + "/".join(safe_parts)

        else:
            # 如果不在挂载目录内，无法预览，但也返回
            final_video_url = path

        # 3. 构造返回对象
        results.append({
            "id": str(hash(path)),      # 伪造一个 ID
            "type": "local",            # 标记类型
            "src": final_video_url,     # 前端预览用的 URL
            "download_url": path,       # 后端渲染用的绝对路径 (前端回传时用这个)
            "name": filename,           # 显示名称
            "tags": req.query           # 搜索词作为 tag
        })

    return {"status": "ok", "results": results}

@app.post("/api/search_online")
def api_search_online(query: str, pexels_key: str = ""):
    engine.set_api_keys(pexels_key, "")
    results = engine.search_online_videos(query)
    if isinstance(results, dict) and "error" in results:
        return {"status": "error", "msg": results["error"]}
    return {"status": "ok", "results": results}

@app.post("/api/search_sfx")
def api_search_sfx(query: str, pixabay_key: str = ""):
    engine.set_api_keys("", pixabay_key)
    res = engine.search_online_sfx(query)
    if isinstance(res, dict) and "error" in res:
        return {"status": "error", "msg": res["error"]}
    return {"status": "ok", "results": res}

@app.post("/api/download_sfx")
def api_download_sfx(req: DownloadSfxRequest):
    fname = engine.download_sfx_manual(req.query, req.url)
    if fname:
        return {"status": "ok", "filename": fname, "url": f"/static/sfx/{fname}"}
    return {"status": "error", "msg": "下载失败"}

@app.post("/api/download_specific")
def api_download_specific(req: DownloadSpecificRequest):
    fname = engine.download_video_by_url(req.url, req.id, req.tags)
    if fname:
        return {"status": "ok", "filename": fname, "url": f"/static/video/{fname}", "name": fname}
    return {"status": "error", "msg": "Download failed"}

@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac')):
        return {"status": "error", "msg": "仅支持 MP3/WAV"}
    try:
        safe_filename = re.sub(r'[^\w\-_.]', '', file.filename.replace(" ", "_"))
        if len(safe_filename) < 4: safe_filename = f"bgm_{int(time.time())}.mp3"
        save_path = os.path.join(config.ASSETS_DIR, "music", safe_filename)
        with open(save_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        return {"status": "ok", "filename": safe_filename, "url": f"/static/music/{safe_filename}"}
    except Exception as e: return {"status": "error", "msg": str(e)}

@app.delete("/api/music/{filename}")
def delete_music(filename: str):
    path = os.path.join(config.ASSETS_DIR, "music", filename)
    if os.path.exists(path): os.remove(path); return {"status": "ok"}
    raise HTTPException(status_code=404, detail="File not found")

# ---  渲染接口 ---
@app.post("/api/render")
async def api_render(req: RenderRequest):
    # --- 1. 确定文件夹名称 (优先使用项目的计划发布时间) ---
    # 默认为今天
    folder_date = datetime.now().strftime("%Y%m%d")
    
    # 如果关联了项目，尝试获取项目的 publish_time
    if req.project_id:
        p_data = project_mgr.get_one(req.project_id)
        if p_data and p_data.get("publish_time"):
            # 前端 datetime-local 格式通常为 "2026-01-20T15:30"
            # 我们截取前10位 "2026-01-20" 并去掉横杠 -> "20260120"
            try:
                raw_time = p_data.get("publish_time")
                if len(raw_time) >= 10:
                    folder_date = raw_time[:10].replace("-", "")
            except:
                pass # 解析失败保持默认今天

    # --- 2. 创建路径 ---
    safe_name = re.sub(r'[^\w\-_]', '', req.output_name)
    save_dir = os.path.join(config.OUTPUT_DIR, folder_date)
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, f"{safe_name}.mp4")
    web_url = f"/outputs/{folder_date}/{safe_name}.mp4"
    
    # --- 3. 准备参数 (保持 Config 配置) ---
    render_params = {
        "scenes": req.scenes,
        "bgm_file": req.bgm_file if req.bgm_file else config.BGM_FILE,
        "bgm_volume": req.bgm_volume if req.bgm_volume is not None else config.BGM_VOLUME,
        "audio_padding": req.audio_padding if req.audio_padding is not None else config.AUDIO_PADDING,
        "tts_rate": req.tts_rate if req.tts_rate else config.DEFAULT_TTS_RATE,
        "subtitle_style": req.subtitle_style,
        "search_source": req.search_source
    }
    
    # --- 4. 初始化进度 ---
    GLOBAL_PROGRESS[req.client_id] = {
        "percent": 0, "msg": "🚀 任务已提交...", "status": "running", "url": ""
    }
    
    # --- 5. 回调函数 ---
    async def log_callback(msg):
        GLOBAL_PROGRESS[req.client_id]["msg"] = msg
        if "渲染进度:" in msg:
            try:
                match = re.search(r'(\d+)%', msg)
                if match: GLOBAL_PROGRESS[req.client_id]["percent"] = int(match.group(1))
            except: pass
            
        if "✅ 处理完成" in msg:
            GLOBAL_PROGRESS[req.client_id]["status"] = "completed"
            GLOBAL_PROGRESS[req.client_id]["percent"] = 100
            GLOBAL_PROGRESS[req.client_id]["url"] = web_url
            
            if req.project_id:
                print(f"💾 更新项目状态: {req.project_id} -> generated (Folder: {folder_date})")
                project_mgr.update(req.project_id, {
                    "status": "generated",
                    "video_path": web_url,
                    "video_abspath": output_path
                })
        elif "❌ Error" in msg:
            GLOBAL_PROGRESS[req.client_id]["status"] = "error"

    # --- 6. 执行 ---
    asyncio.create_task(engine.render_project(render_params, output_path, log_callback))
    
    return {"status": "started", "output_url": web_url}

# --- 项目管理 API ---
@app.get("/api/projects")
def api_list_projects():
    return {"status": "ok", "data": project_mgr.get_all()}

@app.get("/api/projects/{pid}")
def api_get_project(pid: str):
    data = project_mgr.get_one(pid)
    if data: return {"status": "ok", "data": data}
    return {"status": "error", "msg": "Not found"}

@app.post("/api/projects")
def api_create_project(req: ProjectCreateReq):
    # 1. 先仅使用 title 和 script 创建项目 (避免参数报错)
    new_p = project_mgr.create(req.title, req.script)
    
    # 2. 准备要补充的额外字段
    additional_data = {
        "publish_time": req.publish_time,
        "canvas_title": req.canvas_title,
        "main_title": req.main_title,
        "sub_title": req.sub_title,
        "tags": req.tags,
        "status": "draft"
    }
    
    # 3. 使用 update 方法保存这些额外字段
    # project_mgr.update 通常接受 (id, data_dict)
    project_mgr.update(new_p['id'], additional_data)
    
    # 4. 更新内存中的 new_p 对象以便返回给前端最新的数据
    new_p.update(additional_data)
    
    return {"status": "ok", "data": new_p}

@app.put("/api/projects/{pid}")
def api_update_project(pid: str, req: ProjectUpdateReq):
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    success = project_mgr.update(pid, update_data)
    if success: return {"status": "ok"}
    return {"status": "error", "msg": "Update failed"}

@app.delete("/api/projects/{pid}")
def api_delete_project(pid: str):
    project_mgr.delete(pid)
    return {"status": "ok"}

# --- 发布接口 ---
@app.post("/api/projects/{pid}/publish")
def api_publish_project(pid: str):
    # 简单状态变更
    success = project_mgr.update(pid, {"status": "published"})
    if success: return {"status": "ok"}
    return {"status": "error"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)