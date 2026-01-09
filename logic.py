import os
import random
import json
import asyncio
import requests
import ollama
import edge_tts
import re
import glob
import config
import subprocess
import shutil
import traceback
import proglog
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
from openai import OpenAI

#  MoviePy 判断使用内置 FFmpeg
if config.FFMPEG_BINARY and os.path.exists(config.FFMPEG_BINARY):
    os.environ["IMAGEIO_FFMPEG_EXE"] = config.FFMPEG_BINARY
else:
    print(f"⚠️ Warning: FFmpeg path not resolved by config.")

# 修复 PIL
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- 状态更新 Logger ---
# 不再依赖 asyncio loop，直接调用 callback 更新内存
class StatusLogger(proglog.ProgressBarLogger):
    def __init__(self, callback):
        super().__init__(init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0)
        self.callback = callback
    
    def callback(self, **changes):
        for (item, state) in changes.items():
            if not isinstance(state, dict): continue
            total = state.get('total')
            index = state.get('index')
            if total and index:
                percent = int((index / total) * 100)
                if percent % 2 == 0: 
                    # 直接调用回调，传入百分比
                    self.callback(message=f"⏳ 渲染中 ({item})...", percent=percent)

    def message(self, message):
        self.callback(message=f"[MoviePy] {message}")

class WebSocketLogger(proglog.ProgressBarLogger):
    def __init__(self, log_callback, loop):
        super().__init__(init_state=None, bars=None, ignored_bars=None, logged_bars='all', min_time_interval=0, ignore_bars_under=0)
        self.log_callback = log_callback
        self.loop = loop
    
    def callback(self, **changes):
        for (item, state) in changes.items():
            if not isinstance(state, dict): continue
            total = state.get('total')
            index = state.get('index')
            if total and index:
                percent = int((index / total) * 100)
                # 限制频率：每更新 2% 发送一次
                if percent % 2 == 0: 
                    msg = f"⏳ 渲染进度: {percent}%"
                    # 必须使用 run_coroutine_threadsafe，因为这可能是在子线程运行
                    if self.loop and self.loop.is_running():
                         asyncio.run_coroutine_threadsafe(self.log_callback(msg), self.loop)

    def message(self, message):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.log_callback(f"[MoviePy] {message}"), self.loop)

class VideoEngine:
    def __init__(self):
        self.ASSETS_DIR = config.ASSETS_DIR
        self.TEMP_DIR = config.TEMP_DIR
        self.FONT_PATH = config.FONT_PATH
        self.runtime_pexels_key = ""
        self.runtime_pixabay_key = ""
        
        self.llm_provider = config.LLM_PROVIDER
        self.llm_api_key = config.API_KEY
        self.llm_base_url = config.API_BASE_URL
        if self.llm_provider == 'api':
            self.llm_model_name = config.API_MODEL_NAME
        else:
            self.llm_model_name = config.OLLAMA_MODEL
        
        # 目录已经在config.py中创建，这里不再重复创建

    def set_api_keys(self, pexels, pixabay):
        self.runtime_pexels_key = pexels.strip()
        self.runtime_pixabay_key = pixabay.strip()

    def set_llm_config(self, provider, api_key, base_url, model_name):
        self.llm_provider = provider
        if provider == 'api':
            if api_key: self.llm_api_key = api_key
            if base_url: self.llm_base_url = base_url
            if model_name: self.llm_model_name = model_name
        else:
            if model_name: self.llm_model_name = model_name

    def _get_key(self, key_type):
        if key_type == "pexels":
            return self.runtime_pexels_key if self.runtime_pexels_key else config.PEXELS_API_KEY
        elif key_type == "pixabay":
            return self.runtime_pixabay_key if self.runtime_pixabay_key else config.PIXABAY_API_KEY
        return ""

    def sanitize_filename(self, name):
        name = str(name).replace(" ", "_")
        return re.sub(r'[^\w\-_]', '', name)

    def check_ollama_status(self):
        try:
            resp = requests.get("http://127.0.0.1:11434/", timeout=2)
            if resp.status_code == 200: return True
        except: return False
        return False

    def _call_llm(self, prompt):
        print(f"🤖 Calling LLM ({self.llm_provider}): {self.llm_model_name}...")
        if self.llm_provider == 'api':
            if not self.llm_api_key: raise Exception("API Key 未配置")
            client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
            try:
                response = client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"❌ API Error: {e}"); raise e
        else:
            if not self.check_ollama_status(): raise ConnectionError("Ollama 服务未连接")
            try:
                response = ollama.chat(model=self.llm_model_name, messages=[{'role':'user','content':prompt}])
                return response['message']['content']
            except Exception as e:
                print(f"❌ Ollama Error: {e}"); raise e

    def split_text_by_breath(self, text):
        return self.smart_split_text(text)

    def smart_split_text(self, text, min_chars=15):
        text = text.replace("\n", " ").strip()
        if not text: return []
        text = text.replace("...", "@@ELLIPSIS@@")
        sentences = re.split(r'([。！？!?;；]+)', text)
        final_chunks = []
        current_chunk = ""
        for part in sentences:
            part = part.replace("@@ELLIPSIS@@", "...")
            if re.match(r'^[。！？!?;；]+$', part):
                current_chunk += part
                if current_chunk.strip(): final_chunks.append(current_chunk.strip())
                current_chunk = ""
                continue
            temp = current_chunk + part
            if len(temp) < min_chars * 2: current_chunk += part
            else: current_chunk += part
        if current_chunk.strip(): final_chunks.append(current_chunk.strip())
        
        refined = []
        for chunk in final_chunks:
            if len(chunk) > 25:
                commas = re.split(r'([，,])', chunk)
                sub_buf = ""
                for frag in commas:
                    if re.match(r'[，,]', frag):
                        sub_buf += frag
                        if len(sub_buf) > 10: refined.append(sub_buf); sub_buf = ""
                    else: sub_buf += frag
                if sub_buf: refined.append(sub_buf)
            else: refined.append(chunk)
        return [c for c in refined if c.strip()]

    def hex_to_ass_color(self, hex_color):
        hex_color = str(hex_color).lstrip('#')
        if len(hex_color) == 6:
            r, g, b = hex_color[:2], hex_color[2:4], hex_color[4:]
            return f"&H00{b}{g}{r}".upper()
        return "&H00FFFFFF"

    def apply_ass_highlight(self, text, keywords_str, highlight_color_ass, normal_color_ass):
        if not keywords_str: return text
        keywords = [k.strip() for k in re.split(r'[,，]', keywords_str) if k.strip()]
        keywords.sort(key=len, reverse=True)
        final_text = text
        for k in keywords:
            if k in final_text:
                ass_code = f"{{\\c{highlight_color_ass}}}{k}{{\\c{normal_color_ass}}}"
                final_text = final_text.replace(k, ass_code)
        return final_text

    def generate_ass_header(self, style_config):
        norm = style_config.get('normal', {})
        emp = style_config.get('emphasis', {})
        n_size = norm.get('size', 85)
        n_color = self.hex_to_ass_color(norm.get('color', '#FFFFFF'))
        n_outline = self.hex_to_ass_color(norm.get('outline', '#000000'))
        e_size = emp.get('size', 140)
        e_color = self.hex_to_ass_color(emp.get('color', '#FF0000'))
        e_outline = self.hex_to_ass_color(emp.get('outline', '#FFFFFF'))

        # 在ASS格式中，字体名称需要使用完整字体文件名
        font_name = "Alimama ShuHeiTi"

        header = f"""[Script Info]
        Title: Auto Video
        ScriptType: v4.00+
        WrapStyle: 0
        ScaledBorderAndShadow: yes
        YCbCr Matrix: TV.601
        PlayResX: 1920
        PlayResY: 1080

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Normal,{font_name},{n_size},{n_color},{n_color},{n_outline},&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,30,30,60,1
        Style: Emphasis,{font_name},{e_size},{e_color},{e_color},{e_outline},&H00000000,1,0,0,0,100,100,0,0,1,5,0,5,30,30,30,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        """
        return header

    def format_ass_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def clean_for_subtitle(self, text):
        return re.sub(r'[，。！?？,!.、\s]+$', '', text)

    # --- 异步执行 FFmpeg (传递 loop 解决 RuntimeError) ---
    async def run_ffmpeg_async(self, cmd, log_callback, loop):
        if cmd[0] == "ffmpeg":
            if config.FFMPEG_BINARY:
                cmd[0] = config.FFMPEG_BINARY
                # 仅在第一次或调试时打印，避免刷屏
                # print(f"🔧 FFmpeg Cmd: {cmd[0]}") 
            else:
                await log_callback("⚠️ FFmpeg binary not configured!")
        
        # 在线程中运行的同步函数
        def run_sync():
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True, 
                encoding='utf-8',
                errors='ignore'
            )
            buffer = ""
            while True:
                char = process.stderr.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    buffer += char
                    if char in ['\n', '\r']:
                        line = buffer.strip()
                        if line:
                            # 过滤并推送
                            if "frame=" in line or "time=" in line:
                                # [关键] 使用传入的主线程 loop
                                asyncio.run_coroutine_threadsafe(log_callback(f"[FFmpeg] {line}"), loop)
                            elif "Error" in line:
                                asyncio.run_coroutine_threadsafe(log_callback(f"⚠️ {line}"), loop)
                        buffer = ""
            return process.returncode

        # 放到线程池执行
        return await asyncio.to_thread(run_sync)

    # --- 资源功能 (保持不变) ---
    def search_local_videos(self, tag):
        video_dir = os.path.join(self.ASSETS_DIR, "video")
        if not os.path.exists(video_dir): return []
        files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov'))]
        matches = [f for f in files if tag.lower() in f.lower()]
        return matches if matches else files

    def download_video(self, query):
        key = self._get_key("pexels")
        if not key or "粘贴" in key: return None
        headers = {"Authorization": key, "User-Agent": "Mozilla/5.0"}
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=1&orientation=landscape"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            if not data.get('videos'): return None
            vid = data['videos'][0]
            tags = vid.get('tags', [])
            tag_slug = self.sanitize_filename(query)
            extra = "_".join([self.sanitize_filename(t)[:10] for t in tags[:5]])
            fname = f"pexels_{vid['id']}_{tag_slug}_{extra}.mp4"
            if len(fname) > 200: fname = fname[:200] + ".mp4"
            vfiles = vid['video_files']
            target = next((vf['link'] for vf in vfiles if vf['width']==1920), vfiles[0]['link'])
            c = requests.get(target, timeout=60).content
            path = os.path.join(self.ASSETS_DIR, "video", fname)
            with open(path, 'wb') as f: f.write(c)
            return fname
        except: return None

    def download_video_by_url(self, download_url, video_id, tags):
        try:
            tag_slug = self.sanitize_filename(tags)[:50]
            fname = f"pexels_{video_id}_{tag_slug}.mp4"
            path = os.path.join(self.ASSETS_DIR, "video", fname)
            if os.path.exists(path): return fname
            headers = {"User-Agent": "Mozilla/5.0"}
            c = requests.get(download_url, headers=headers, timeout=120).content
            with open(path, 'wb') as f: f.write(c)
            return fname
        except: return None

    def search_online_videos(self, query):
        key = self._get_key("pexels")
        if not key or "粘贴" in key: return {"error": "API Key Missing"}
        headers = {"Authorization": key, "User-Agent": "Mozilla/5.0"}
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=12&orientation=landscape"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            results = []
            for v in data.get('videos', []):
                preview = v['video_files'][0]['link']
                for f in v['video_files']:
                    if 600 <= f.get('width', 0) <= 1280: preview = f['link']; break
                download = v['video_files'][0]['link']
                for f in v['video_files']:
                    if f.get('width', 0) == 1920: download = f['link']; break
                tags_str = query
                if v.get('tags'): tags_str = v['tags'][0]
                results.append({
                    "type": "online", "id": v['id'], "src": preview,
                    "download_url": download, "tags": tags_str, "name": f"Pexels ID: {v['id']}"
                })
            return results
        except Exception as e: return {"error": str(e)}

    def search_online_sfx(self, query):
        url = "https://www.myinstants.com/api/v1/instants/"
        params = {"name": query, "format": "json"}
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=5)
            data = r.json()
            results = []
            for item in data.get('results', [])[:15]:
                sound = item.get('sound')
                if not sound: continue
                name = item.get('name', 'SFX').replace("'", "").replace('"', '')
                results.append({
                    "id": str(random.randint(10000, 99999)),
                    "name": name,
                    "duration": 2,
                    "download_url": sound, "preview_url": sound
                })
            return results
        except: return {"error": "SFX Error"}

    def download_sfx_manual(self, query, download_url=None):
        sfx_dir = os.path.join(self.ASSETS_DIR, "sfx")
        if download_url:
            try:
                base = f"auto_{self.sanitize_filename(query)}"
                fname = f"{base}.mp3"
                path = os.path.join(sfx_dir, fname)
                if not os.path.exists(path):
                    c = requests.get(download_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).content
                    with open(path, 'wb') as f: f.write(c)
                return fname
            except: return None
        else:
            return self.get_dynamic_sfx(query, sfx_dir)

    def get_dynamic_sfx(self, search_term, save_dir):
        if not search_term: return None
        local = [f for f in os.listdir(save_dir) if not f.startswith('.')]
        for f in local:
            if search_term.lower() in f.lower(): return os.path.join(save_dir, f)
        
        fallback = {
            "whoosh": "https://assets.mixkit.co/active_storage/sfx/2568/2568-preview.mp3",
            "ding": "https://assets.mixkit.co/active_storage/sfx/961/961-preview.mp3",
            "boom": "https://assets.mixkit.co/active_storage/sfx/3004/3004-preview.mp3",
            "keyboard": "https://assets.mixkit.co/active_storage/sfx/238/238-preview.mp3"
        }
        dl = None
        for k,v in fallback.items():
            if k in search_term.lower(): dl=v; break
        if dl:
            try:
                c = requests.get(dl, headers={'User-Agent':'Mozilla/5.0'}).content
                p = os.path.join(save_dir, f"auto_{self.sanitize_filename(search_term)}.mp3")
                with open(p, 'wb') as f: f.write(c)
                return p
            except: pass
        return None

    def load_sfx_resource(self, sfx_query, sfx_dir):
        if not sfx_query: return None
        exact = os.path.join(sfx_dir, sfx_query)
        if os.path.exists(exact): return exact
        local = [f for f in os.listdir(sfx_dir) if not f.startswith('.')]
        for f in local:
            if sfx_query.lower() in f.lower(): return os.path.join(sfx_dir, f)
        return None

    def get_video_clip_safe(self, video_info, duration, log_callback=None):
        video_path = None
        if isinstance(video_info, str):
            v_type = "random" if video_info == "random" else "local"
            v_name = video_info
            v_dl_url = ""
            v_tags = ""
            v_id = ""
        else:
            v_type = video_info.get('type', 'local')
            v_name = video_info.get('name', '')
            v_dl_url = video_info.get('download_url', '')
            v_tags = video_info.get('tags', '')
            v_id = video_info.get('id', 'temp')

        if v_type == 'online':
            safe_tag = self.sanitize_filename(v_tags)
            fname = f"pexels_{v_id}_{safe_tag}.mp4"
            local_path = os.path.join(self.ASSETS_DIR, "video", fname)
            if os.path.exists(local_path):
                video_path = local_path
            else:
                try:
                    c = requests.get(v_dl_url, timeout=60).content
                    with open(local_path, 'wb') as f: f.write(c)
                    video_path = local_path
                except: video_path = None
        elif v_type == 'local':
            if v_name and v_name != 'random':
                p = os.path.join(self.ASSETS_DIR, "video", v_name)
                if os.path.exists(p): video_path = p

        if not video_path:
            v_dir = os.path.join(self.ASSETS_DIR, "video")
            v_files = [f for f in os.listdir(v_dir) if f.endswith('.mp4')]
            if v_files: video_path = os.path.join(v_dir, random.choice(v_files))

        if not video_path:
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

        try:
            vc = VideoFileClip(video_path)
            if vc.duration < duration:
                vc = vc.loop(duration=duration)
            else:
                max_s = max(0, vc.duration - duration - 0.1)
                s = random.uniform(0, max_s)
                vc = vc.subclip(s, s+duration)
            vc = vc.resize(height=1080)
            if vc.w > 1920: vc = vc.crop(x1=vc.w/2-960, width=1920, height=1080)
            elif vc.w < 1920: vc = vc.resize(width=1920).crop(x1=0, y1=vc.h/2-540, width=1920, height=1080)
            return vc
        except:
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

    def analyze_script(self, text):
        print(f"🤖 Calling LLM: {self.llm_model_name}...")
        prompt = f"""
        你是一个视频脚本专家。分析文案，返回严格 JSON 列表。
        
        【规则】
        1. **visual_tags**: 必须是**英文**单词！用于搜索视频。(如: "business man", "future city", "ai robot")。
           - 必须根据每一句话的具体意象联想。
        2. text: 严禁修改原文。
        3. keywords: 1-3个中文重点词，逗号分隔。
        4. is_emphasis: 标题/金句为true。
        5. sfx_search: 音效(whoosh/ding/boom)。
        
        格式:
        [
          {{"text": "原文", "keywords": "重点", "is_emphasis": false, "visual_tags": ["tag_en"], "sfx_search": ""}}
        ]
        Text: {text}
        """
        try:
            content = self._call_llm(prompt)
            clean_content = re.sub(r'```json\s*', '', content)
            clean_content = re.sub(r'```', '', clean_content).strip()
            s = clean_content.find('[')
            e = clean_content.rfind(']')+1
            scenes = json.loads(clean_content[s:e])
            
            used_identifiers = set()

            for scene in scenes:
                kw = scene.get('keywords')
                if isinstance(kw, list): scene['keywords'] = ", ".join([str(k) for k in kw])
                elif kw is None: scene['keywords'] = ""
                else: scene['keywords'] = str(kw)

                tags = scene.get('visual_tags', [])
                scene['voice'] = config.DEFAULT_VOICE
                scene['video_info'] = {"type": "local", "src": "", "name": "random"} 
                
                if tags:
                    search_query = tags[0]
                    online_info = self.search_online_videos(search_query)
                    selected_vid = None
                    if isinstance(online_info, list) and online_info:
                        for vid in online_info:
                            if str(vid['id']) not in used_identifiers:
                                selected_vid = vid
                                used_identifiers.add(str(vid['id']))
                                break
                        if not selected_vid: selected_vid = random.choice(online_info)
                        scene['video_info'] = selected_vid
                    else:
                        local = self.search_local_videos(search_query)
                        if local:
                            random.shuffle(local)
                            target = None
                            for f in local:
                                if f not in used_identifiers:
                                    target = f
                                    used_identifiers.add(f)
                                    break
                            if not target: target = random.choice(local)
                            scene['video_info'] = {"type": "local", "name":target, "src":f"/static/video/{target}"}
            return scenes
        except Exception as e:
            print(f"LLM Error: {e}")
            return [{"text": text, "visual_tags": ["abstract"], "keywords":"", "sfx_search":"", "is_emphasis": False, "video_info": {"type":"local","name":"random"}}]

    # --- 渲染核心 ---
    async def render_project(self, params, output_file, log_callback=None):
        # 获取主线程 loop，用于子线程回调
        loop = asyncio.get_running_loop()
        
        async def log(msg):
            print(msg)
            # 关键：执行 server.py 传进来的回调
            if log_callback: await log_callback(msg)

        try:
            await log("🎬 初始化渲染引擎...")
            scene_data = params['scenes']
            bgm_file = params.get('bgm_file', '')
            bgm_vol = float(params.get('bgm_volume', 0.1))
            global_padding = float(params.get('audio_padding', 0.2)) 
            tts_rate = params.get('tts_rate', config.DEFAULT_TTS_RATE)
            sub_style = params.get('subtitle_style', {})

            json_file = output_file.replace('.mp4', '.json')
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(params, f, ensure_ascii=False, indent=2)
                await log(f"💾 项目文件已保存")
            except: pass

            norm_style = sub_style.get('normal', {})
            emp_style = sub_style.get('emphasis', {})
            ass_color_normal = self.hex_to_ass_color(norm_style.get('color', '#FFFFFF'))
            ass_color_highlight = self.hex_to_ass_color(emp_style.get('color', '#FF0000'))

            for f in glob.glob(os.path.join(self.TEMP_DIR, "scene_*.mp4")): os.remove(f)
            
            scene_files = []
            subtitles_events = []
            current_time = 0.0
            total_scenes = len(scene_data)

            # [关键] 传入 loop 到 Logger
            custom_logger = WebSocketLogger(log_callback, loop)

            for idx, scene in enumerate(scene_data):
                text = scene['text']
                voice = scene.get('voice', config.DEFAULT_VOICE)
                sfx = scene.get('sfx_search', '')
                is_emphasis = scene.get('is_emphasis', False)
                keywords = scene.get('keywords', '').strip()
                scene_padding = float(scene.get('audio_padding', global_padding))
                video_info = scene.get('video_info', {})

                await log(f"🔨 处理分镜 {idx+1}/{total_scenes}...")
                
                raw_text_clean = re.sub(r'[\(（].*?[\)）]', '', text).strip()
                if not raw_text_clean: continue

                sub_chunks = self.smart_split_text(raw_text_clean, min_chars=15)
                scene_audio_clips = []
                scene_total_duration = 0.0
                
                for sub_idx, chunk in enumerate(sub_chunks):
                    tts_text = chunk.strip()
                    if not tts_text: continue
                    tpath = os.path.join(self.TEMP_DIR, f"tts_{idx}_{sub_idx}.mp3")
                    
                    tts_success = False
                    for _ in range(3):
                        try:
                            await edge_tts.Communicate(tts_text, voice, rate=tts_rate).save(tpath)
                            if os.path.exists(tpath) and os.path.getsize(tpath) > 100:
                                tts_success = True; break
                        except: await asyncio.sleep(1)
                    
                    if not tts_success:
                        ac = AudioArrayClip(np.zeros((44100, 2)), fps=44100).set_duration(1.0)
                    else:
                        ac = AudioFileClip(tpath)

                    chunk_dur = ac.duration + scene_padding
                    scene_audio_clips.append(ac.set_start(scene_total_duration))
                    
                    start_s = self.format_ass_time(current_time + scene_total_duration)
                    end_s = self.format_ass_time(current_time + scene_total_duration + chunk_dur)
                    disp = self.clean_for_subtitle(chunk)
                    
                    if is_emphasis:
                        content = disp
                        ass_l = f"Dialogue: 0,{start_s},{end_s},Emphasis,,0,0,0,,{{\\fad(50,0)}}{content}"
                        subtitles_events.append(ass_l)
                    else:
                        if disp:
                            final_t = disp
                            if keywords: final_t = self.apply_ass_highlight(disp, keywords, ass_color_highlight, ass_color_normal)
                            ass_l = f"Dialogue: 0,{start_s},{end_s},Normal,,0,0,0,,{{\\fad(80,80)}}{final_t}"
                            subtitles_events.append(ass_l)
                    
                    scene_total_duration += chunk_dur

                if not scene_audio_clips:
                    scene_total_duration = 2.0
                    combined_audio = AudioArrayClip(np.zeros((int(44100*2), 2)), fps=44100)
                else:
                    combined_audio = CompositeAudioClip(scene_audio_clips).set_duration(scene_total_duration)
                
                final_audio = [combined_audio]
                
                if sfx:
                    sp = self.download_sfx_manual(sfx)
                    if sp:
                         sp_path = os.path.join(self.ASSETS_DIR, "sfx", sp)
                         try: 
                             af = AudioFileClip(sp_path).volumex(0.6).set_start(0)
                             final_audio.append(af)
                             await log(f"   🔊 添加音效: {os.path.basename(sp)}")
                         except: pass
                
                vc = self.get_video_clip_safe(video_info, scene_total_duration, log)
                vc = vc.set_audio(CompositeAudioClip(final_audio))
                
                scene_out = os.path.join(self.TEMP_DIR, f"scene_{idx}.mov")
                
                temp_audio_abs_path = os.path.join(self.TEMP_DIR, f"temp_{idx}.wav")

                # --- [关键修复] 线程化写入 & 传递 loop ---
                def write_segment():
                    vc.write_videofile(
                        scene_out, 
                        fps=30, 
                        preset="ultrafast", 
                        logger=custom_logger, 
                        codec="libx264", 
                        audio_codec="pcm_s16le", 
                        temp_audiofile=temp_audio_abs_path, 
                        remove_temp=True
                    )
                await asyncio.to_thread(write_segment)
                
                vc.close()
                combined_audio.close()
                scene_files.append(scene_out)
                current_time += scene_total_duration

            if not scene_files: return False
            await log("🔗 缝合视频...")
            list_path = os.path.join(self.TEMP_DIR, "concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in scene_files: f.write(f"file '{os.path.abspath(p)}'\n")
            temp_concat = os.path.join(self.TEMP_DIR, "temp_concat.mov")
            
            # [关键修复] 传递 loop 给 ffmpeg_async
            await self.run_ffmpeg_async(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", temp_concat], log_callback, loop)
            
            await log("🎵 混合BGM...")
            final_clip = VideoFileClip(temp_concat)
            if bgm_file:
                bp = os.path.join(self.ASSETS_DIR, "music", bgm_file)
                if os.path.exists(bp):
                    try:
                        bgm = AudioFileClip(bp).volumex(bgm_vol)
                        from moviepy.audio.fx.all import audio_loop
                        bgm = audio_loop(bgm, duration=final_clip.duration)
                        final_clip = final_clip.set_audio(CompositeAudioClip([final_clip.audio, bgm]) if final_clip.audio else bgm)
                    except: pass
            
            temp_video_bgm = os.path.join(self.TEMP_DIR, "temp_with_bgm.mov")
            
            def write_bgm():
                temp_final_wav_path = os.path.join(self.TEMP_DIR, "temp_final.wav")
                final_clip.write_videofile(temp_video_bgm, fps=30, codec="libx264", audio_codec="pcm_s16le", temp_audiofile=temp_final_wav_path, remove_temp=True, logger=custom_logger)
            await asyncio.to_thread(write_bgm)
            
            final_clip.close()

            await log("📝 压制字幕与最终输出...")
            
            # 1. 写入 ASS 文件
            ass_str = self.generate_ass_header(sub_style)
            for l in subtitles_events: ass_str += l + "\n"
            
            # 获取绝对路径，防止 FFmpeg 找不到
            ass_p = os.path.abspath(os.path.join(self.TEMP_DIR, "s.ass"))
            with open(ass_p, "w", encoding="utf-8") as f: f.write(ass_str)
            
            # 2. 准备字体路径
            fdir = os.path.abspath(os.path.join(self.ASSETS_DIR, "fonts"))
            font_p = os.path.join(fdir, "font.ttf")
            
            # --- [核心兼容性修复函数] ---
            def format_ffmpeg_path(path):
                # 1. 转绝对路径
                # 2. 将 Windows 的反斜杠 \ 替换为 /
                return os.path.abspath(path).replace("\\", "/")

            # 处理所有路径
            safe_ass_p = format_ffmpeg_path(ass_p)
            safe_fdir = format_ffmpeg_path(fdir)
            safe_input = format_ffmpeg_path(temp_video_bgm)
            safe_output = format_ffmpeg_path(output_file)
            
            # --- [关键] 构造滤镜字符串 ---
            # Windows 必须加单引号 '' 包裹路径，否则盘符冒号(C:)会被识别为分隔符
            if os.path.exists(font_p):
                vf = f"ass='{safe_ass_p}':fontsdir='{safe_fdir}'"
            else:
                # 字体不存在时的回退
                await log("⚠️ 未检测到 assets/fonts/font.ttf，将使用系统回退字体。")
                vf = f"ass='{safe_ass_p}'"
            
            # 3. 执行 FFmpeg
            # 注意：-i 和 输出路径 不需要加引号，subprocess 会处理；
            # 但 -vf 内部的路径必须加引号（上面已经加了）
            await self.run_ffmpeg_async([
                "ffmpeg", "-y", 
                "-i", safe_input, 
                "-vf", vf, 
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", 
                "-c:a", "aac", "-b:a", "192k", 
                safe_output
            ], log_callback, loop)
            
            # 4. 清理与完成
            try:
                os.remove(temp_video_bgm); os.remove(temp_concat); os.remove(list_path); os.remove(ass_p)
                shutil.rmtree(self.TEMP_DIR); os.makedirs(self.TEMP_DIR, exist_ok=True)
            except: pass
            
            final_filename = os.path.basename(output_file)
            # 注意：这里的 URL 是给前端用的，保持 web 路径格式 /outputs/...
            final_url = f"/outputs/{final_filename}"
            
            await log(f"✅ 处理完成@@@{final_url}")
            return True

        except Exception as e:
            await log(f"❌ Error: {traceback.format_exc()}")
            return False