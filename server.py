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
from logic import VideoEngine

app = FastAPI()
engine = VideoEngine()

app.mount("/static", StaticFiles(directory="assets"), name="static")
app.mount("/temp", StaticFiles(directory="temp_web"), name="temp")
if not os.path.exists(config.OUTPUT_DIR): os.makedirs(config.OUTPUT_DIR)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

templates = Jinja2Templates(directory="templates")

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

class AnalyzeRequest(BaseModel):
    text: str
    pexels_key: str = ""
    pixabay_key: str = ""

class RenderRequest(BaseModel):
    client_id: str
    scenes: list
    output_name: str
    bgm_file: str = ""
    bgm_volume: float = 0.1
    audio_padding: float = 0.2
    tts_rate: str = "+15%" # [新增]
    subtitle_style: dict = {}

class DownloadSpecificRequest(BaseModel):
    url: str
    id: str
    tags: str

@app.get("/")
def index(request: Request):
    v_dir = os.path.join(config.ASSETS_DIR, "video")
    m_dir = os.path.join(config.ASSETS_DIR, "music")
    s_dir = os.path.join(config.ASSETS_DIR, "sfx")
    
    videos = [f for f in os.listdir(v_dir) if f.endswith(('.mp4', '.mov'))]
    music = [f for f in os.listdir(m_dir) if f.endswith(('.mp3', '.wav'))]
    sfx = [f for f in os.listdir(s_dir) if f.endswith(('.mp3', '.wav'))]
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "videos": videos,
        "music": music,
        "sfx_list": sfx,
        "voice_options": config.VOICE_OPTIONS,
        "default_voice": config.DEFAULT_VOICE,
        "default_tts_rate": config.DEFAULT_TTS_RATE,
        "default_pexels": config.PEXELS_API_KEY if "粘贴" not in config.PEXELS_API_KEY else "",
        "default_pixabay": config.PIXABAY_API_KEY if "粘贴" not in config.PIXABAY_API_KEY else ""
    })

@app.get("/api/history")
def get_history():
    files = glob.glob(os.path.join(config.OUTPUT_DIR, "*.mp4"))
    files.sort(key=os.path.getmtime, reverse=True)
    history = []
    for f in files:
        name = os.path.basename(f)
        size_mb = round(os.path.getsize(f) / (1024*1024), 1)
        ctime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        # 检查是否有对应的 json
        json_name = name.replace('.mp4', '.json')
        has_json = os.path.exists(os.path.join(config.OUTPUT_DIR, json_name))
        
        history.append({
            "name": name, 
            "size": size_mb, 
            "time": ctime, 
            "url": f"/outputs/{name}",
            "has_project": has_json,
            "json_file": json_name
        })
    return {"status": "ok", "history": history}

# [新增] 读取项目数据
@app.get("/api/history/load/{filename}")
def load_project_data(filename: str):
    path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"status": "ok", "data": data}
    raise HTTPException(status_code=404, detail="Project file not found")

@app.delete("/api/history/{filename}")
def delete_history(filename: str):
    path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        # 尝试删除配套 JSON
        json_path = path.replace('.mp4', '.json')
        if os.path.exists(json_path):
            os.remove(json_path)
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="File not found")

@app.websocket("/ws/logs/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    await manager.connect(client_id)
    try:
        while True:
            log_msg = await manager.active_connections[client_id].get()
            await websocket.send_text(log_msg)
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    engine.set_api_keys(req.pexels_key, req.pixabay_key)
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
    elif results is None:
        return {"status": "error", "msg": "No results"}
    return {"status": "ok", "results": results}

@app.post("/api/search_sfx")
def api_search_sfx(query: str, pixabay_key: str = ""):
    engine.set_api_keys("", pixabay_key)
    # 调用 logic 中的真实搜索
    results = engine.search_online_sfx(query)
    
    if isinstance(results, dict) and "error" in results:
        return {"status": "error", "msg": results["error"]}
    return {"status": "ok", "results": results}

# [更新] 音效下载 (接收 URL 参数)
class DownloadSfxRequest(BaseModel):
    query: str
    url: str = "" # 可选参数

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

# 音乐上传接口
@app.post("/api/music/upload")
async def upload_music(file: UploadFile = File(...)):
    # 允许的扩展名
    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac')):
        return {"status": "error", "msg": "仅支持 MP3/WAV/FLAC 格式"}
    
    try:
        # 清洗文件名 (防止中文乱码或路径攻击)
        safe_filename = re.sub(r'[^\w\-_.]', '', file.filename.replace(" ", "_"))
        # 如果文件名全被洗掉了，给个默认名
        if len(safe_filename) < 4: 
            safe_filename = f"uploaded_bgm_{int(time.time())}.mp3"
            
        save_path = os.path.join(config.ASSETS_DIR, "music", safe_filename)
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "ok", 
            "filename": safe_filename, 
            "url": f"/static/music/{safe_filename}"
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# 音乐删除接口
@app.delete("/api/music/{filename}")
def delete_music(filename: str):
    path = os.path.join(config.ASSETS_DIR, "music", filename)
    if os.path.exists(path):
        os.remove(path)
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/render")
async def api_render(req: RenderRequest):
    safe_name = re.sub(r'[^\w\-_]', '', req.output_name)
    output_path = os.path.join(config.OUTPUT_DIR, f"{safe_name}.mp4")
    
    render_params = {
        "scenes": req.scenes,
        "bgm_file": req.bgm_file,
        "bgm_volume": req.bgm_volume,
        "audio_padding": req.audio_padding,
        "tts_rate": req.tts_rate, # [新增]
        "subtitle_style": req.subtitle_style
    }
    
    async def log_callback(msg):
        await manager.send_log(req.client_id, msg)

    asyncio.create_task(engine.render_project(render_params, output_path, log_callback))
    return {"status": "started", "output_url": f"/outputs/{safe_name}.mp4"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)