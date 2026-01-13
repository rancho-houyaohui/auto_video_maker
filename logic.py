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

# 强制 MoviePy 使用内置 FFmpeg
if os.path.exists(config.FFMPEG_BINARY):
    os.environ["IMAGEIO_FFMPEG_EXE"] = config.FFMPEG_BINARY

# 修复 PIL
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- 线程安全的 WebSocket Logger ---
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
                if percent % 5 == 0: 
                    msg = f"⏳ 渲染进度: {percent}%"
                    asyncio.run_coroutine_threadsafe(self.log_callback(msg), self.loop)

    def message(self, message):
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
        
        for d in ["video", "sfx", "music", "fonts", "outputs"]: 
            os.makedirs(os.path.join(self.ASSETS_DIR, d), exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

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
        # 默认最大宽度 18 个字 (超过就尝试在逗号处切开，如果没逗号，也会保留完整句子)
        return self.smart_split_text(text, max_chars=18)

    # --- [核心优化] 智能贪婪合并分段算法 ---
    def smart_split_text(self, text, max_chars=18):
        """
        策略：
        1. 优先合并：尽可能把短句拼成长句，直到超过 max_chars。
        2. 逻辑断句：遇到强结束符(。？！)必须断开。
        3. 弱断句：遇到逗号时，如果当前缓冲区已经够长了，就断开；否则继续拼。
        """
        text = text.replace("\n", " ").strip()
        # 剥离包裹符号
        text = text.strip('"').strip("'").strip('“').strip('”').strip('(').strip(')').strip('（').strip('）')
        if not text: return []
        
        text = text.replace("...", "@@ELLIPSIS@@")
        
        # 1. 先按所有标点切成“原子” (Atom)，保留标点
        # 例如: "你好，世界。" -> ["你好", "，", "世界", "。"]
        atoms = re.split(r'([，。！?？,!.、;；：:]+)', text)
        
        # 2. 重组原子为带标点的片段
        # -> ["你好，", "世界。"]
        segments = []
        current_segment = ""
        for item in atoms:
            if not item: continue
            current_segment += item
            # 如果 item 包含标点，说明这个 segment 结束了
            if re.search(r'[，。！?？,!.、;；：:]+', item):
                segments.append(current_segment)
                current_segment = ""
        if current_segment: segments.append(current_segment)
        
        # 3. 贪婪合并逻辑
        final_chunks = []
        current_buffer = ""
        
        for seg in segments:
            # 还原省略号
            seg = seg.replace("@@ELLIPSIS@@", "...")
            
            # 判断是否包含强结束符 (句号/问号/感叹号)
            is_strong_end = bool(re.search(r'[。！？!?]', seg))
            
            # 预测合并后的长度
            # 如果 (当前缓存 + 新片段) 超过限制，且当前缓存不为空 -> 先结算当前缓存
            if len(current_buffer) + len(seg) > max_chars:
                if current_buffer:
                    final_chunks.append(current_buffer.strip())
                    current_buffer = ""
            
            current_buffer += seg
            
            # 如果遇到强结束符，必须立刻结算 (不跨句合并)
            if is_strong_end:
                final_chunks.append(current_buffer.strip())
                current_buffer = ""
                
        # 结算剩余
        if current_buffer.strip():
            final_chunks.append(current_buffer.strip())
            
        # 4. 二次校验：如果某一段太长且没有标点 (极少见)，强制切分 (防止溢出屏幕)
        # 这里设置为 25 字强切
        refined_chunks = []
        for chunk in final_chunks:
            if len(chunk) > 25:
                 # 暴力对半切
                 mid = len(chunk) // 2
                 refined_chunks.append(chunk[:mid])
                 refined_chunks.append(chunk[mid:])
            else:
                refined_chunks.append(chunk)

        # 5. 过滤掉纯标点的无效片段
        return [c for c in refined_chunks if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', c)]

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
        n_size = norm.get('size', 100)
        n_color = self.hex_to_ass_color(norm.get('color', '#FFFFFF'))
        n_outline = self.hex_to_ass_color(norm.get('outline', '#000000'))
        e_size = emp.get('size', 180)
        e_color = self.hex_to_ass_color(emp.get('color', '#FF0000'))
        e_outline = self.hex_to_ass_color(emp.get('outline', '#FFFFFF'))

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
        Style: Normal,Arial,{n_size},{n_color},{n_color},{n_outline},&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,30,30,450,1
        Style: Emphasis,Arial,{e_size},{e_color},{e_color},{e_outline},&H00000000,1,0,0,0,100,100,0,0,1,6,0,5,30,30,350,1
        Style: Yellow,Arial,{e_size},{e_color},{e_color},{e_outline},&H00000000,1,0,0,0,100,100,0,0,1,5,0,5,30,30,350,1

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
        return re.sub(r'[，。！?？,!.、;；：:\"\'“”\[\]【】]+', ' ', text).strip()

    async def run_ffmpeg_async(self, cmd, log_callback, loop):
        if cmd[0] == "ffmpeg":
            if os.path.exists(config.FFMPEG_BINARY):
                cmd[0] = config.FFMPEG_BINARY
        
        def run_sync():
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, encoding='utf-8', errors='ignore'
            )
            buffer = ""
            while True:
                char = process.stderr.read(1)
                if not char and process.poll() is not None: break
                if char:
                    buffer += char
                    if char in ['\n', '\r']:
                        line = buffer.strip()
                        if line:
                            if "frame=" in line or "time=" in line:
                                asyncio.run_coroutine_threadsafe(log_callback(f"[FFmpeg] {line}"), loop)
                            elif "Error" in line:
                                asyncio.run_coroutine_threadsafe(log_callback(f"⚠️ {line}"), loop)
                        buffer = ""
            return process.returncode

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
            target = next((vf['link'] for vf in video_files if vf['width']==1920), vfiles[0]['link'])
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

    # --- 音效搜索 (MyInstants) ---
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
                    "name": name, "duration": 2, "download_url": sound, "preview_url": sound
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
            # 压暗
            vc = vc.fx(vfx.colorx, 0.7)
            vc = vc.without_audio()

            if vc.duration < duration:
                vc = vc.loop(duration=duration)
            else:
                max_s = max(0, vc.duration - duration - 0.1)
                s = random.uniform(0, max_s)
                vc = vc.subclip(s, s+duration)
            
            vc = vc.set_duration(duration)
            vc = vc.resize(height=1080)
            if vc.w > 1920: vc = vc.crop(x1=vc.w/2-960, width=1920, height=1080)
            elif vc.w < 1920: vc = vc.resize(width=1920).crop(x1=0, y1=vc.h/2-540, width=1920, height=1080)
            return vc
        except:
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

    def analyze_script(self, text):
        print(f"🤖 Calling LLM: {self.llm_model_name}...")
        
        # 1. Python 先拆好
        text_clean_for_split = re.sub(r'[\(（].*?[\)）]', '', text).strip()
        
        # [核心优化 1] 分词逻辑入口
        pre_split_segments = self.smart_split_text(text_clean_for_split, max_chars=18)
        
        if not pre_split_segments: return []
        segments_json = json.dumps(pre_split_segments, ensure_ascii=False)

        prompt = f"""
        你是一个JSON转换器。任务是为输入的每一句文案匹配**英文画面关键词**和**音效**。
        
        【输入数据】
        {segments_json}
        
        【输出要求】
        1. "visual_tags": 提取句子中的实体名词，翻译成 1-2 个**英文单词** (用于搜视频)。不要用抽象词。
        2. "keywords": 提取句子中 1-3 个字的**中文重点词** (用于字幕高亮)。
        3. "sfx_search": 音效。**只有**当句子包含"震惊、转折、强调、疑问"语气时才填 (whoosh/ding/boom/pop)，**普通叙述请留空字符串**，不要每句都填！
        4. "is_emphasis": 是否为标题或金句 (true/false)。
        5. "sfx_search": 对于金句选择音效，仅限从以下选择: whoosh, ding, boom, keyboard, pop。
        
        【输出示例】
        [
          {{"text": "第一句文案", "keywords": "重点", "is_emphasis": false, "visual_tags": ["tag1", "tag2"], "sfx_search": "whoosh"}},
          {{"text": "第二句文案", "keywords": "核心", "is_emphasis": true, "visual_tags": ["tag3", "tag4"], "sfx_search": "boom"}}
        ]
        """
        try:
            content = self._call_llm(prompt)
            clean_content = re.sub(r'```json\s*', '', content)
            clean_content = re.sub(r'```', '', clean_content).strip()
            s = clean_content.find('[')
            e = clean_content.rfind(']')+1
            scenes = json.loads(clean_content[s:e])
            
            used_identifiers = set()
            final_scenes = []
            
            for i, original_text in enumerate(pre_split_segments):
                scene_data = {}
                if i < len(scenes):
                    scene_data = scenes[i]
                else:
                    scene_data = {"visual_tags": ["abstract"], "keywords": "", "sfx_search": ""}
                
                # 强制回填原文
                scene_data['text'] = original_text
                
                kw = scene_data.get('keywords')
                if isinstance(kw, list): scene_data['keywords'] = ", ".join([str(k) for k in kw])
                elif kw is None: scene_data['keywords'] = ""
                else: scene_data['keywords'] = str(kw)

                # 默认不加音效
                if 'sfx_search' not in scene_data: scene_data['sfx_search'] = ""

                tags = scene_data.get('visual_tags', [])
                scene_data['voice'] = config.DEFAULT_VOICE
                scene_data['video_info'] = {"type": "local", "src": "", "name": "random"} 
                
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
                        scene_data['video_info'] = selected_vid
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
                            scene_data['video_info'] = {"type": "local", "name":target, "src":f"/static/video/{target}"}
                
                final_scenes.append(scene_data)

            return final_scenes
        except Exception as e:
            print(f"LLM Error: {e}")
            fallback_scenes = []
            for t in pre_split_segments:
                fallback_scenes.append({
                    "text": t, "visual_tags": ["abstract"], "keywords":"", "sfx_search":"", 
                    "is_emphasis": False, "video_info": {"type":"local","name":"random"}
                })
            return fallback_scenes

    # --- 渲染核心 ---
    async def render_project(self, params, output_file, log_callback=None):
        loop = asyncio.get_running_loop()
        
        async def log(msg):
            print(msg)
            if log_callback: await log_callback(msg)

        try:
            await log("🎬 初始化渲染引擎...")
            scene_data = params['scenes']
            bgm_file = params.get('bgm_file', '')
            bgm_vol = float(params.get('bgm_volume', 0.1))
            global_padding = float(params.get('audio_padding', 0)) 
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

                sub_chunks = self.smart_split_text(raw_text_clean, max_chars=15)
                scene_audio_clips = []
                scene_total_duration = 0.0
                
                for sub_idx, chunk in enumerate(sub_chunks):
                    tts_text = chunk.strip()
                    
                    if not tts_text or not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', tts_text):
                        continue

                    tpath = os.path.join(self.TEMP_DIR, f"tts_{idx}_{sub_idx}.mp3")
                    
                    tts_success = False
                    for _ in range(3):
                        try:
                            await edge_tts.Communicate(tts_text, voice, rate=tts_rate).save(tpath)
                            if os.path.exists(tpath) and os.path.getsize(tpath) > 1024:
                                tts_success = True; break
                        except: await asyncio.sleep(1)
                    
                    if not tts_success:
                        ac = AudioArrayClip(np.zeros((44100, 2)), fps=44100).set_duration(0.1)
                        await log(f"⚠️ TTS失败，跳过: {tts_text[:5]}")
                    else:
                        try:
                            ac = AudioFileClip(tpath).volumex(1.5)
                        except:
                            ac = AudioArrayClip(np.zeros((44100, 2)), fps=44100).set_duration(0.1)

                    chunk_dur = ac.duration + scene_padding
                    scene_audio_clips.append(ac.set_start(scene_total_duration))
                    
                    start_s = self.format_ass_time(current_time + scene_total_duration)
                    end_s = self.format_ass_time(current_time + scene_total_duration + chunk_dur)
                    
                    disp = self.clean_for_subtitle(chunk)
                    
                    if disp:
                        if is_emphasis:
                            # 1. 霸屏模式
                            content = disp
                            # 如果字数太多，自动换行
                            if len(content) > 8:
                                content = content[:8] + "\\N" + content[8:]
                                
                            ass_l = f"Dialogue: 1,{start_s},{end_s},Emphasis,,0,0,0,,{{\\fad(50,0)}}{content}"
                            subtitles_events.append(ass_l)
                        else:
                            # 2. 普通模式 (双轨)
                            ass_bottom = f"Dialogue: 1,{start_s},{end_s},Normal,,0,0,0,,{{\\fad(80,80)}}{disp}"
                            subtitles_events.append(ass_bottom)
                            
                            # if keywords:
                            #     kws = [k.strip() for k in re.split(r'[,，]', keywords) if k.strip()]
                            #     if kws:
                            #         kw_str = "\\N".join(kws) if len(kws)>1 or sum(len(k) for k in kws)>8 else "  ".join(kws)
                            #         ass_center = f"Dialogue: 1,{start_s},{end_s},Emphasis,,0,0,0,,{{\\fad(50,50)}}{kw_str}"
                            #         subtitles_events.append(ass_center)
                    
                    scene_total_duration += chunk_dur

                if not scene_audio_clips:
                    await log(f"⚠️ 跳过无效分镜 {idx+1}")
                    continue

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
                
                # --- [核心修复：物理截取视频] ---
                # 1. 确保视频被物理截断到指定时长
                # get_video_clip_safe 内部虽然有 subclip，但为了保险，这里再次强制执行
                # 如果视频比音频长，强制截取前 scene_total_duration 秒
                if vc.duration > scene_total_duration:
                    # 随机找一个起点 (或者从头开始)
                    # 注意：get_video_clip_safe 已经做过随机了，这里直接截取 0 到 end 即可
                    vc = vc.subclip(0, scene_total_duration)
                elif vc.duration < scene_total_duration:
                    # 如果视频不够长，循环播放
                    vc = vc.loop(duration=scene_total_duration)
                
                # 2. 再次显式设置 duration (双重保险)
                vc = vc.set_duration(scene_total_duration)
                
                # 3. 移除原声 (防止素材自带声音干扰)
                vc = vc.without_audio()
                
                # 4. 合成音频
                final_audio_clip = CompositeAudioClip(final_audio).set_duration(scene_total_duration)
                vc = vc.set_audio(final_audio_clip)
                
                # 5. 写入文件
                scene_out = os.path.join(self.TEMP_DIR, f"scene_{idx}.mov")
                
                def write_segment():
                    vc.write_videofile(
                        scene_out, 
                        fps=30, 
                        preset="ultrafast", 
                        logger=custom_logger, 
                        codec="libx264", 
                        audio_codec="pcm_s16le", 
                        temp_audiofile=f"{self.TEMP_DIR}/temp_{idx}.wav", 
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
                for p in scene_files:
                    safe_p = os.path.abspath(p).replace("\\", "/")
                    f.write(f"file '{safe_p}'\n")

            temp_concat = os.path.join(self.TEMP_DIR, "temp_concat.mov")
            
            def format_p(path): return os.path.abspath(path).replace("\\", "/")
            safe_concat = format_p(temp_concat)
            safe_list = format_p(list_path)

            await self.run_ffmpeg_async(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", safe_list, "-c", "copy", safe_concat], log_callback, loop)
            
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
                final_clip.write_videofile(temp_video_bgm, fps=30, codec="libx264", audio_codec="pcm_s16le", temp_audiofile="temp_final.wav", remove_temp=True, logger=custom_logger)
            
            await asyncio.to_thread(write_bgm)
            final_clip.close()

            await log("📝 压制字幕与最终输出...")
            ass_str = self.generate_ass_header(sub_style)
            for l in subtitles_events: ass_str += l + "\n"
            
            ass_p = os.path.abspath(os.path.join(self.TEMP_DIR, "s.ass"))
            with open(ass_p, "w", encoding="utf-8") as f: f.write(ass_str)
            
            fdir = os.path.abspath(os.path.join(self.ASSETS_DIR, "fonts"))
            font_p = os.path.join(fdir, "font.ttf")
            
            safe_ass = format_p(ass_p)
            safe_fdir = format_p(fdir)
            safe_in = format_p(temp_video_bgm)
            safe_out = format_p(output_file)
            
            vf = f"ass='{safe_ass}':fontsdir='{safe_fdir}'" if os.path.exists(font_p) else f"ass='{safe_ass}'"
            
            await self.run_ffmpeg_async([
                "ffmpeg", "-y", 
                "-i", safe_in, 
                "-vf", vf, 
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", 
                "-c:a", "aac", "-b:a", "192k", 
                safe_out
            ], log_callback, loop)
            
            os.remove(temp_video_bgm); os.remove(temp_concat); os.remove(list_path); os.remove(ass_p)
            shutil.rmtree(self.TEMP_DIR); os.makedirs(self.TEMP_DIR, exist_ok=True)
            
            final_filename = os.path.basename(output_file)
            final_url = f"/outputs/{final_filename}"
            await log(f"✅ 处理完成@@@{final_url}")
            return True

        except Exception as e:
            await log(f"❌ Error: {traceback.format_exc()}")
            return False