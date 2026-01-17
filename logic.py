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
        return self.smart_split_text(text, max_chars=30)

    # --- [核心优化] 智能贪婪合并分段算法 ---
    def smart_split_text(self, text, max_chars=30):
        """
        优化版：增加对破折号的支持，移除暴力切分，改用软截断。
        """
        text = text.replace("\n", " ").strip()
        text = text.strip('"').strip("'").strip('“').strip('”').strip('(').strip(')').strip('（').strip('）')
        if not text: return []
        
        text = text.replace("...", "@@ELLIPSIS@@")
        
        # 1. 补充支持破折号 —— 和空格作为切分点
        atoms = re.split(r'([，。！?？,!.、;；：:——\s]+)', text)
        
        segments = []
        current_segment = ""
        for item in atoms:
            if not item: continue
            current_segment += item
            if re.search(r'[，。！?？,!.、;；：:——\s]+', item):
                segments.append(current_segment)
                current_segment = ""
        if current_segment: segments.append(current_segment)
        
        final_chunks = []
        current_buffer = ""
        
        for seg in segments:
            seg = seg.replace("@@ELLIPSIS@@", "...")
            is_strong_end = bool(re.search(r'[。！？!?]', seg))
            
            # 2. 只有当 buffer 确实太长，且新片段加上去会显著过长时才切分
            # 如果当前 buffer 字数很少（例如<5），即使加上去超标了也尽量不切，防止孤儿词
            if len(current_buffer) + len(seg) > max_chars:
                if len(current_buffer) > 5: 
                    final_chunks.append(current_buffer.strip())
                    current_buffer = ""
            
            current_buffer += seg
            
            if is_strong_end:
                final_chunks.append(current_buffer.strip())
                current_buffer = ""
                
        if current_buffer.strip():
            final_chunks.append(current_buffer.strip())
            
        # 3. 彻底删除原有的 Step 4 (暴力对半切)，防止“陷阱”被切成“陷”“阱”
        # 改为最后一道过滤
        return [c for c in final_chunks if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', c)]

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
        import concurrent.futures

        # 1. 预处理：先分句
        text_clean_for_split = re.sub(r'[\(（].*?[\)）]', '', text).strip()
        pre_split_segments = self.smart_split_text(text_clean_for_split, max_chars=30)
        if not pre_split_segments: return []

        total_segments = len(pre_split_segments)
        print(f"📝 总计 {total_segments} 个分镜，准备分批并行处理...")

        # --- 配置 ---
        BATCH_SIZE = 8  # 每批处理 8 句 (根据显存/API限制调整，推荐 5-10)
        
        # 如果是本地 Ollama，并发设为 1 避免显存爆炸；如果是 API (OpenAI/DeepSeek等)，可以设为 3-5
        MAX_WORKERS = 1 if self.llm_provider == 'ollama' else 4 

        # 结果容器，预先占位，保证顺序
        final_results = [None] * total_segments

        # 定义单个批次的处理函数
        def process_batch(batch_data):
            batch_index, segment_chunk = batch_data
            
            # 构造仅包含文本的列表供 LLM 分析
            chunk_json = json.dumps(segment_chunk, ensure_ascii=False)

            # 优化后的 Prompt：明确要求不返回原文，减少生成时间
            prompt = f"""
            你是一个视频脚本分析师。请分析输入的文案列表。
            
            【输入数据】
            {chunk_json}
            
            【任务】
            按顺序为每一句生成 JSON 对象，**不要返回原文(text字段)**，仅返回分析属性。
            
            【属性要求】
            1. "v": (visual_tags) 提取句子中的实体名词，翻译成 1-2 个**英文单词** (用于搜视频)。不要用抽象词。
            2. "k": (keywords) 提取句子中 1-3 个字的**中文重点词** (用于字幕高亮)。
            3. "s": (sfx) 音效。**只有**当句子包含"震惊、转折、强调、疑问"语气时才填 ( whoosh, ding, boom, keyboard, pop)，**普通叙述请留空字符串**，不要每句都填！
            4. "e": (is_emphasis) Boolean，是否为金句/标题 (true/false)。
            
            【输出格式】
            严格的 JSON 列表，不需要 Markdown 标记，例如：
            [
              {{"v": ["city", "night"], "k": "夜景", "s": "whoosh", "e": false}},
              {{"v": ["money"], "k": "赚钱", "s": "ding", "e": true}}
            ]
            """
            
            try:
                # 调用 LLM
                response = self._call_llm(prompt)
                
                # 清洗 JSON
                clean_content = re.sub(r'```json\s*', '', response)
                clean_content = re.sub(r'```', '', clean_content).strip()
                # 尝试提取列表部分
                s = clean_content.find('[')
                e = clean_content.rfind(']') + 1
                if s != -1 and e != -1:
                    clean_content = clean_content[s:e]
                
                return batch_index, json.loads(clean_content)
            except Exception as e:
                print(f"⚠️ Batch {batch_index} Error: {e}")
                return batch_index, []

        # 2. 准备批次数据
        batches = []
        for i in range(0, total_segments, BATCH_SIZE):
            chunk = pre_split_segments[i : i + BATCH_SIZE]
            batches.append((i, chunk))

        # 3. 并行/串行执行
        # 使用 ThreadPoolExecutor 进行并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(process_batch, b): b for b in batches}
            
            for future in concurrent.futures.as_completed(future_to_batch):
                start_idx, result_list = future.result()
                
                # 回填数据
                chunk_len = len(batches[start_idx // BATCH_SIZE][1])
                
                for offset in range(chunk_len):
                    abs_index = start_idx + offset
                    original_text = pre_split_segments[abs_index]
                    
                    # 默认兜底数据
                    scene_data = {
                        "text": original_text,
                        "visual_tags": ["abstract"],
                        "keywords": "",
                        "sfx_search": "",
                        "is_emphasis": False
                    }
                    
                    # 尝试读取 LLM 结果
                    if result_list and offset < len(result_list):
                        item = result_list[offset]
                        # 映射简写字段回完整字段
                        scene_data["visual_tags"] = item.get("v", ["abstract"])
                        scene_data["keywords"] = item.get("k", "")
                        scene_data["sfx_search"] = item.get("s", "")
                        scene_data["is_emphasis"] = item.get("e", False)
                    
                    final_results[abs_index] = scene_data

        # 4. 后处理：资源匹配 (搜图/搜视频)
        # 这部分逻辑保持不变，但移到了最后统一处理
        print("🔍 正在匹配视频素材...")
        used_identifiers = set()
        final_scenes = []

        for scene_data in final_results:
            # 防止前面的并发错误导致 None
            if scene_data is None: continue 
            
            # 处理 keywords 格式
            kw = scene_data.get('keywords')
            if isinstance(kw, list): scene_data['keywords'] = ", ".join([str(k) for k in kw])
            elif kw is None: scene_data['keywords'] = ""
            else: scene_data['keywords'] = str(kw)

            # 默认参数
            scene_data['voice'] = config.DEFAULT_VOICE
            scene_data['video_info'] = {"type": "local", "src": "", "name": "random"} 
            
            tags = scene_data.get('visual_tags', [])
            
            # --- 资源搜索逻辑 (复用原有的逻辑) ---
            if tags:
                search_query = tags[0]
                # 1. 搜在线
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
                    # 2. 搜本地
                    local = self.search_local_videos(search_query)
                    if local:
                        target = None
                        random.shuffle(local)
                        for f in local:
                            if f not in used_identifiers:
                                target = f
                                used_identifiers.add(f)
                                break
                        if not target: target = random.choice(local)
                        scene_data['video_info'] = {"type": "local", "name":target, "src":f"/static/video/{target}"}
            
            final_scenes.append(scene_data)

        return final_scenes

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
                            # 1. 霸屏模式优化：不再暴力切断，而是智能换行
                            content = disp
                            
                            # 策略：如果有标点（顿号、逗号），优先在标点后换行
                            if "、" in content:
                                content = content.replace("、", "、\\N")
                            elif "，" in content:
                                content = content.replace("，", "，\\N")
                            elif len(content) > 12: 
                                # 只有真的非常长且没标点时，才在中间换行
                                mid = len(content) // 2
                                content = content[:mid] + "\\N" + content[mid:]
                                
                            # 移除末尾可能多余的换行符
                            if content.endswith("\\N"): content = content[:-2]

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