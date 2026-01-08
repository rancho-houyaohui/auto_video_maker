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
from logic import VideoEngine
from project_manager import project_mgr

app = FastAPI()

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

class RenderRequest(BaseModel):
    client_id: str
    scenes: list
    output_name: str
    bgm_file: str = ""
    bgm_volume: float = 0.1
    audio_padding: float = 0.2
    tts_rate: str = "+15%" 
    subtitle_style: dict = {}

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
    publish_time: str = ""

class ProjectUpdateReq(BaseModel):
    title: str = None
    script: str = None
    video_path: str = None
    cover_path: str = None
    status: str = None
    publish_time: str = None
    scenes_data: list = None 

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
    files = glob.glob(os.path.join(config.OUTPUT_DIR, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    history = []
    for f in files:
        name = os.path.basename(f)
        size_mb = round(os.path.getsize(f) / (1024*1024), 1)
        ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        json_name = name.replace('.mp4', '.json')
        has_json = os.path.exists(os.path.join(config.OUTPUT_DIR, json_name))
        history.append({
            "name": name, "size": size_mb, "time": ctime, 
            "url": f"/outputs/{name}", "has_project": has_json, "json_file": json_name
        })
    return {"status": "ok", "history": history}

@app.get("/api/history/load/{filename}")
def load_project_data(filename: str):
    path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        return {"status": "ok", "data": data}
    raise HTTPException(status_code=404, detail="Project file not found")

@app.delete("/api/history/{filename}")
def delete_history(filename: str):
    path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        json_path = path.replace('.mp4', '.json')
        if os.path.exists(json_path): os.remove(json_path)
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="File not found")

# --- 核心业务 API ---
@app.websocket("/ws/logs/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await manager.connect(client_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    engine.set_api_keys(req.pexels_key, req.pixabay_key)
    engine.set_llm_config(req.llm_provider, req.llm_api_key, req.llm_base_url, req.llm_model)
    scenes = engine.analyze_script(req.text)
    for scene in scenes:
        tags = scene.get('visual_tags', [])
        matches = []
        if tags: matches = engine.search_local_videos(tags[0])
        scene['video'] = matches[0] if matches else "random" 
        scene['voice'] = config.DEFAULT_VOICE 
    return {"status": "ok", "scenes": scenes}

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
    safe_name = re.sub(r'[^\w\-_]', '', req.output_name)
    output_path = os.path.join(config.OUTPUT_DIR, f"{safe_name}.mp4")
    
    render_params = {
        "scenes": req.scenes,
        "bgm_file": req.bgm_file,
        "bgm_volume": req.bgm_volume,
        "audio_padding": req.audio_padding,
        "tts_rate": req.tts_rate,
        "subtitle_style": req.subtitle_style
    }
    
    # 1. 初始化全局进度状态
    GLOBAL_PROGRESS[req.client_id] = {
        "percent": 0,
        "msg": "🚀 任务已提交...",
        "status": "running",
        "url": ""
    }
    
    # 2. 定义回调函数：直接更新内存字典
    async def log_callback(msg):
        # print(f"[{req.client_id}] {msg}") # 可选：在终端打印日志方便调试
        
        # 更新消息
        GLOBAL_PROGRESS[req.client_id]["msg"] = msg
        
        # 尝试解析进度百分比 (格式: "⏳ 渲染进度: 45%")
        if "渲染进度:" in msg:
            try:
                # 提取数字
                match = re.search(r'(\d+)%', msg)
                if match:
                    GLOBAL_PROGRESS[req.client_id]["percent"] = int(match.group(1))
            except: pass
            
        # 检查完成状态
        if "✅ 处理完成" in msg:
            GLOBAL_PROGRESS[req.client_id]["status"] = "completed"
            GLOBAL_PROGRESS[req.client_id]["percent"] = 100
            if "@@@" in msg:
                GLOBAL_PROGRESS[req.client_id]["url"] = msg.split("@@@")[1]
        
        # 检查错误状态
        elif "❌ Error" in msg:
            GLOBAL_PROGRESS[req.client_id]["status"] = "error"

    # 3. 启动后台任务
    asyncio.create_task(engine.render_project(render_params, output_path, log_callback))
    
    return {"status": "started", "output_url": f"/outputs/{safe_name}.mp4"}

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
    new_p = project_mgr.create(req.title, req.script, req.publish_time)
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

# 注意：移除了 /api/projects/{pid}/publish_now 接口

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)