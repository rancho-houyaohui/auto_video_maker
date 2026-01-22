import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import clip
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
from vfx_core import VisualEffects

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

        # === [新增] 向量数据库初始化 ===
        self.vector_index_path = os.path.join(config.ASSETS_DIR, "vector_index.pt") # 假设你的索引在这里
        self.clip_model_path = "ViT-B-32.pt" # 你的本地模型路径
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clip_model = None
        self.clip_preprocess = None
        self.vector_db = None
        
    def _load_vector_resources(self):
        """懒加载：只有用到向量搜索时才加载，节省内存"""
        if self.clip_model is None:
            print(f"🧠 Loading CLIP model...")
            if os.path.exists(self.clip_model_path):
                self.clip_model, self.clip_preprocess = clip.load(self.clip_model_path, device=self.device)
            else:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            
        if self.vector_db is None:
            if os.path.exists(self.vector_index_path):
                print(f"📂 Loading Vector Index...")
                self.vector_db = torch.load(self.vector_index_path)
                # 预加载向量到显存/内存加速计算
                if len(self.vector_db) > 0:
                    self.db_vectors = torch.cat([item['vector'] for item in self.vector_db]).to(self.device)
            else:
                print("⚠️ Vector index not found.")
                self.vector_db = []

    async def generate_tts_audio(self, text, voice, output_path, engine="edge", sovits_url=None):
        """
        统一的语音生成接口
        engine: 'edge' | 'sovits'
        """
        # 1. GPT-SoVITS 逻辑
        if engine == "sovits" and sovits_url:
            try:
                # 这里假设 SoVITS 兼容标准 API 格式，根据你实际部署的情况调整
                payload = {
                    "text": text,
                    "text_language": "zh"
                }
                # 注意：requests 是同步的，建议用 aiohttp，这里为了简单演示用 requests
                # 实际生产中最好放入线程池
                def _run_request():
                    return requests.post(f"{sovits_url}/tts", json=payload, timeout=10)
                
                resp = await asyncio.to_thread(_run_request)
                
                if resp.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    return True
            except Exception as e:
                print(f"❌ SoVITS Error: {e}, falling back to Edge-TTS")
        
        # 2. Edge-TTS 逻辑 (默认)
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"❌ Edge-TTS Error: {e}")
            return False
    # --- 从向量库随机获取一个没用过的视频 ---
    def _get_random_vector_clip(self, exclude_set):
        """
        兜底逻辑：当搜索不到素材，或者素材不够填满时间时，
        从库里随机捞一个没用过的视频。
        """
        self._load_vector_resources()
        if not self.vector_db: return None
        
        # 尝试 50 次寻找未使用的
        for _ in range(50):
            item = random.choice(self.vector_db)
            path = item['path']
            if os.path.exists(path) and path not in exclude_set:
                return path
        
        # 如果实在找不到（库太小），只能勉强复用一个
        item = random.choice(self.vector_db)
        return item['path'] if os.path.exists(item['path']) else None

    def search_vector_match(self, query, top_k=5, exclude_set=None):
        """
        返回 Top K 个结果，且不在 exclude_set 中
        """
        self._load_vector_resources()
        if not self.vector_db or self.db_vectors is None: return []
        if exclude_set is None: exclude_set = set()

        with torch.no_grad():
            text_input = clip.tokenize([query]).to(self.device)
            text_features = self.clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * text_features @ self.db_vectors.T).softmax(dim=-1)
            values, indices = similarity[0].topk(min(top_k * 2, len(self.vector_db))) # 多取一点方便过滤
            
        results = []
        for i in range(len(indices)):
            idx = indices[i].item()
            item = self.vector_db[idx]
            path = item['path']
            
            # 核心：过滤掉不存在的文件 和 全局已使用的文件
            if os.path.exists(path) and path not in exclude_set:
                results.append(path)
                if len(results) >= top_k: break
                
        return results

    def get_video_clip_smart(self, video_info, duration, use_vector=True, use_pexels=True):
        """
        智能获取视频素材：
        1. 本地文件名精确匹配
        2. 向量数据库模糊匹配 (新增)
        3. Pexels 在线搜索
        4. 兜底逻辑
        """
        # 解析 video_info (兼容前端传来的各种格式)
        search_query = ""
        if isinstance(video_info, dict):
            if video_info.get('type') == 'local' and os.path.exists(video_info.get('src', '')):
                return VideoFileClip(video_info['src']) # 绝对路径直接返回
            search_query = video_info.get('tags', '') or video_info.get('name', '')
        elif isinstance(video_info, str):
            search_query = video_info

        # A. 向量搜索 (如果开启)
        if use_vector and search_query:
            matches = self.search_vector_match(search_query)
            if matches:
                # 策略：如果有多个匹配，且时长不够，这里可以做多镜头拼接 (参考之前的逻辑)
                # 这里简单处理：随机选一个匹配度高的
                best_match = matches[0]
                print(f"✅ Vector Match: {search_query} -> {os.path.basename(best_match)}")
                return self._process_clip(best_match, duration)

        # B. Pexels 在线搜索 (如果开启)
        if use_pexels and search_query:
            # 复用你原有的 download_video 逻辑，这里简化演示
            # pexels_file = self.download_video(search_query) 
            # if pexels_file: ...
            pass 

        # C. 兜底 (随机本地)
        # ... (保留你原有的兜底逻辑) ...
        return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

    def get_dynamic_visuals_smart(self, candidate_paths, target_duration, global_used_set, effect_config=None):
        """
        effect_config: JSON中的 visual_effect 字段
        """
        clips = []
        current_duration = 0.0
        candidate_queue = list(candidate_paths)
        
        # 默认特效配置
        if not effect_config: 
            effect_config = {"camera": "Zoom_In_Slow", "filter": "None"}

        while current_duration < target_duration:
            selected_path = None
            
            # A. 优先从候选池拿
            if candidate_queue:
                selected_path = candidate_queue.pop(0)
                if selected_path in global_used_set: selected_path = None
            
            # B. 随机兜底
            if not selected_path:
                selected_path = self._get_random_vector_clip(global_used_set)
                if not selected_path: break

            try:
                global_used_set.add(selected_path)
                
                # 1. 加载基础视频
                clip = VideoFileClip(selected_path).without_audio()
                
                # 2. 【绝对静态】的尺寸预处理 (Reset to 1080p)
                # 这一步保证进入 VFX 之前，视频是标准的 1920x1080 整数尺寸
                target_h = 1080
                target_w = 1920
                
                if clip.h != target_h: 
                    clip = clip.resize(height=target_h)
                
                if clip.w > target_w:
                    # 静态居中裁剪
                    x_c = clip.w // 2
                    half_w = target_w // 2
                    clip = clip.crop(x1=x_c - half_w, width=target_w, height=target_h)
                elif clip.w < target_w:
                    clip = clip.resize(width=target_w)
                    y_c = clip.h // 2
                    half_h = target_h // 2
                    clip = clip.crop(y1=y_c - half_h, width=target_w, height=target_h)
                
                # -----------------------------------------------------------
                # 3. [应用 Plan B 特效] 
                # 调用新的 vfx_core (OpenCV版)
                
                from vfx_core import VisualEffects
                
                camera_move = effect_config.get("camera", "Zoom_In_Slow")
                filter_type = effect_config.get("filter", "None")
                
                # 先滤镜
                clip = VisualEffects.apply_filter(clip, filter_type)
                # 后运镜 (现在是像素级处理，不涉及元数据，绝对安全)
                clip = VisualEffects.apply_camera_movement(clip, camera_move)
                # -----------------------------------------------------------
                
                clips.append(clip)
                current_duration += clip.duration
                
            except Exception as e:
                print(f"⚠️ Clip Error {selected_path}: {e}")
                continue

        # 兜底：如果没生成任何片段
        if not clips:
            print("⚠️ No clips valid, returning black screen.")
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=target_duration)

        # 4. 拼接
        # method="compose" 比 "chain" 更稳健，能处理不同属性的视频
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # 5. 时间修剪
        if final_clip.duration >= target_duration:
            final_clip = final_clip.subclip(0, target_duration)
        else:
            final_clip = final_clip.loop(duration=target_duration)
            
        return final_clip

    def _process_clip(self, path, duration):
        """统一的素材处理：静音、循环、裁剪"""
        try:
            vc = VideoFileClip(path).without_audio()
            if vc.duration < duration:
                vc = vc.loop(duration=duration)
            else:
                vc = vc.subclip(0, duration)
            return vc.resize(height=1080).crop(x1=vc.w/2-960, width=1920, height=1080)
        except:
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

    def parse_direct_json(self, json_data):
        scenes = []
        timeline = json_data.get('timeline', [])
        
        print("🔍 Pre-scanning vector database for frontend preview...")
        
        for block in timeline:
            # 提取关键词
            visual_tags = block.get('visual_search_queries', [])
            # 提取特效配置
            visual_effect = block.get('visual_effect', {})
            
            # 提取音效名 (用于后续查找)
            sfx_name = block.get('center_highlight', {}).get('sfx', '')

            # === [新增] 预搜索视频 ===
            # 这里只搜 Top 1，用于给前端展示“由于使用了这个关键词，匹配到了这个视频”
            # 实际渲染时会搜 Top 5 进行填充，这里只是预览
            preview_video = "random"
            if visual_tags:
                # 使用第一个关键词去搜
                matches = self.search_vector_match(visual_tags[0], top_k=1)
                if matches:
                    # 返回给前端绝对路径，或者基于 server.py 的相对 URL
                    # 假设前端能通过文件协议或静态服务访问
                    preview_video = matches[0] 

            scene = {
                "text": block['sentence_text'],
                "visual_tags": visual_tags, 
                "video": preview_video, # 前端拿到这个字段就可以显示了
                
                # 将特效配置传递下去
                "visual_effect": visual_effect,
                
                "keywords": block.get('center_highlight', {}).get('text', ''),
                "is_emphasis": block.get('center_highlight', {}).get('enabled', False),
                "sfx": sfx_name,
                
                "voice": config.DEFAULT_VOICE,
                "audio_padding": 0.2
            }
            scenes.append(scene)
            
        return scenes

    def _get_random_vector_clip(self, exclude_set):
        """
        兜底逻辑：当搜索不到素材，或者素材不够填满时间时，
        从库里随机捞一个没用过的视频。
        """
        self._load_vector_resources()
        if not self.vector_db: return None
        
        # 尝试 50 次寻找未使用的
        for _ in range(50):
            item = random.choice(self.vector_db)
            path = item['path']
            if os.path.exists(path) and path not in exclude_set:
                return path
        
        # 如果实在找不到（库太小），只能勉强复用一个
        item = random.choice(self.vector_db)
        return item['path'] if os.path.exists(item['path']) else None


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

    # --- 音效查找：支持模糊匹配 ---
    def _find_local_sfx(self, keyword):
        """
        在 assets/sfx 目录下模糊查找包含 keyword 的文件
        例如 keyword="Boom", 可以匹配到 "Impact_Boom_01.mp3"
        """
        if not keyword: return None
        sfx_dir = os.path.join(self.ASSETS_DIR, "sfx")
        if not os.path.exists(sfx_dir): return None
        
        # 获取所有音频文件
        files = [f for f in os.listdir(sfx_dir) if f.lower().endswith(('.mp3', '.wav', '.aac', '.m4a'))]
        
        keyword = keyword.lower().strip()
        
        # 1. 优先：文件名以 keyword 开头 (例如 "Boom_01.mp3")
        for f in files:
            if f.lower().startswith(keyword):
                return os.path.join(sfx_dir, f)

        # 2. 次选：文件名包含 keyword (例如 "Cinematic_Boom.mp3")
        for f in files:
            if keyword in f.lower():
                return os.path.join(sfx_dir, f)
        
        return None

    def download_sfx_manual(self, query, download_url=None):
        # 1. 先尝试本地模糊搜索
        local_path = self._find_local_sfx(query)
        if local_path:
            print(f"   🔊 Local SFX Found: '{query}' -> {os.path.basename(local_path)}")
            return os.path.basename(local_path) # 返回文件名即可，后续逻辑会处理路径

        # 2. 如果没找到且有 URL，尝试下载 (保留原有逻辑)
        if download_url:
            return self.get_dynamic_sfx(query, os.path.join(self.ASSETS_DIR, "sfx"), download_url=download_url)
            
        # 3. 都没有，尝试去在线库匹配 (MyInstants等，保留原有逻辑)
        return self.get_dynamic_sfx(query, os.path.join(self.ASSETS_DIR, "sfx"))

    def get_dynamic_sfx(self, search_term, save_dir, download_url=None):
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

        if download_url:
            try:
                base = f"auto_{self.sanitize_filename(search_term)}.mp3"
                path = os.path.join(save_dir, base)
                if not os.path.exists(path):
                    c = requests.get(download_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15).content
                    with open(path, 'wb') as f: f.write(c)
                return base
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

    def analyze_script(self, text, search_source="vector"):
        """
        文本转分镜
        search_source: 'vector' (优先本地) | 'pexels' (只用在线)
        """
        print(f"🤖 Calling LLM: {self.llm_model_name}...")
        import concurrent.futures

        # 1. 预处理：分句
        text_clean_for_split = re.sub(r'[\(（].*?[\)）]', '', text).strip()
        pre_split_segments = self.smart_split_text(text_clean_for_split, max_chars=30)
        if not pre_split_segments: return []

        total_segments = len(pre_split_segments)
        print(f"📝 总计 {total_segments} 个分镜，准备分析...")

        # 配置并发
        BATCH_SIZE = 8
        MAX_WORKERS = 1 if self.llm_provider == 'ollama' else 4 
        final_results = [None] * total_segments

        def process_batch(batch_data):
            batch_index, segment_chunk = batch_data
            chunk_json = json.dumps(segment_chunk, ensure_ascii=False)

            # === [核心修改] 提示词升级：增加 'fx' (VFX) 字段 ===
            if search_source == 'vector':
                fx_prompt = """
                **具象主体** (适合Pexels): 具体的动作或物体，如 "man running", "clock ticking", "fist hitting table"。
                """
            else:
                fx_prompt = """
                **情绪氛围** (适合向量库): 抽象的感觉，如 "anxiety", "loneliness", "oppression", "chaos"。
                """

            prompt = f"""
            你是一个视频脚本导演。请分析输入的文案列表。
            
            【输入数据】
            {chunk_json}
            
            【任务】
            按顺序为每一句生成 JSON 对象。
            
            【属性要求】
            1. "v": (visual_tags) 返回一个包含 2-3 个英文短语的数组 (Array)，必须覆盖以下维度以提高匹配率：
               {fx_prompt}
               - 例如: ["man sitting alone", "solitude", "dark noir style"]
            2. "k": (keywords) 提取 1-3 个字的中文重点词 (用于字幕高亮)。
            3. "s": (sfx) 音效。可选 [Boom, Whoosh, Ding, Keyboard, Glass_Shatter, Silence]。无则留空。
            4. "e": (is_emphasis) Boolean，是否为金句 (true/false)。
            5. "fx": (visual_effect) 运镜指令。根据句子情绪选择：
               - "Zoom_In_Slow" (默认，通用)
               - "Zoom_In_Fast" (强调、震惊)
               - "Pan_Right" (叙述、过程)
               - "Shake" (焦虑、混乱、痛苦)
               - "Slow_Mo" (史诗、总结、唯美)
            
            【输出格式】
            纯 JSON 列表，无 Markdown：
            [
              {{"v": ["city", "night"], "k": "夜景", "s": "whoosh", "e": false, "fx": "Pan_Right"}},
              {{"v": ["fist", "punch"], "k": "反击", "s": "boom", "e": true, "fx": "Zoom_In_Fast"}}
            ]
            """
            
            try:
                response = self._call_llm(prompt)
                # 清洗 JSON
                clean_content = re.sub(r'```json\s*', '', response)
                clean_content = re.sub(r'```', '', clean_content).strip()
                s = clean_content.find('[')
                e = clean_content.rfind(']') + 1
                if s != -1 and e != -1:
                    clean_content = clean_content[s:e]
                return batch_index, json.loads(clean_content)
            except Exception as e:
                print(f"⚠️ Batch {batch_index} Error: {e}")
                return batch_index, []

        # 执行 LLM 分析
        batches = []
        for i in range(0, total_segments, BATCH_SIZE):
            chunk = pre_split_segments[i : i + BATCH_SIZE]
            batches.append((i, chunk))

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {executor.submit(process_batch, b): b for b in batches}
            for future in concurrent.futures.as_completed(future_to_batch):
                start_idx, result_list = future.result()
                chunk_len = len(batches[start_idx // BATCH_SIZE][1])
                
                for offset in range(chunk_len):
                    abs_index = start_idx + offset
                    original_text = pre_split_segments[abs_index]
                    
                    # 默认数据
                    scene_data = {
                        "text": original_text,
                        "visual_tags": ["abstract"],
                        "keywords": "",
                        "sfx_search": "",
                        "is_emphasis": False,
                        "visual_effect": {"camera": "Zoom_In_Slow", "filter": "None"} # 默认特效
                    }
                    
                    if result_list and offset < len(result_list):
                        item = result_list[offset]
                        scene_data["visual_tags"] = item.get("v", ["abstract"])
                        scene_data["keywords"] = item.get("k", "")
                        scene_data["sfx_search"] = item.get("s", "")
                        scene_data["is_emphasis"] = item.get("e", False)
                        
                        # [新增] 解析运镜指令
                        fx_code = item.get("fx", "Zoom_In_Slow")
                        # 简单的逻辑：如果是 Shake，滤镜加个 High_Contrast 增加氛围
                        filter_code = "High_Contrast" if fx_code == "Shake" else "None"
                        scene_data["visual_effect"] = {"camera": fx_code, "filter": filter_code}
                    
                    final_results[abs_index] = scene_data

        # 4. 后处理：资源匹配 (根据 search_source 参数)
        print(f"🔍 正在匹配视频素材 (Mode: {search_source})...")
        used_identifiers = set()
        final_scenes = []

        for scene_data in final_results:
            if scene_data is None: continue 
            
            # 格式化 keywords
            kw = scene_data.get('keywords')
            if isinstance(kw, list): scene_data['keywords'] = ", ".join([str(k) for k in kw])
            elif kw is None: scene_data['keywords'] = ""
            else: scene_data['keywords'] = str(kw)

            scene_data['voice'] = config.DEFAULT_VOICE
            scene_data['video_info'] = {"type": "smart_search", "tags": []} 
            
            tags = scene_data.get('visual_tags', [])
            if isinstance(tags, str): tags = [tags]
            scene_data['video_info']['tags'] = tags

            match_found = False
            main_tag = tags[0] if tags else "abstract"

            # ================= [核心修改] 搜索分支控制 =================
            
            # --- 分支 A: 优先向量搜索 (默认) ---
            if search_source == "vector":
                vector_matches = self.search_vector_match(main_tag, top_k=1)
                if vector_matches:
                    best_path = vector_matches[0]
                    scene_data['video_info'] = {
                        "type": "local", 
                        "src": best_path, 
                        "name": os.path.basename(best_path),
                        "tags": tags 
                    }
                    print(f"   ✅ Vector Hit: {main_tag} -> {os.path.basename(best_path)}")
                    match_found = True
            
            # --- 分支 B: Pexels 搜索 (作为 fallback 或 指定模式) ---
            # 如果模式是 'pexels'，或者模式是 'vector' 但没搜到
            if not match_found and (search_source == "pexels" or search_source == "vector"):
                # 如果是 vector 模式没搜到，打印日志
                if search_source == "vector":
                    print(f"   🌐 Vector Miss, trying Pexels: {main_tag}")
                
                online_info = self.search_online_videos(main_tag)
                if isinstance(online_info, list) and online_info:
                    selected_vid = None
                    for vid in online_info:
                        if str(vid['id']) not in used_identifiers:
                            selected_vid = vid
                            used_identifiers.add(str(vid['id']))
                            break
                    if not selected_vid: selected_vid = random.choice(online_info)
                    scene_data['video_info'] = selected_vid
                    match_found = True

            # 3. 兜底
            if not match_found:
                scene_data['video_info'] = {"type": "random", "tags": tags}

            final_scenes.append(scene_data)

        return final_scenes

    # --- 渲染核心 ---
    async def render_project(self, params, output_file, log_callback=None):
        # 全局去重集合，贯穿整个视频
        global_used_paths = set() 
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
            search_source = params.get('search_source', 'vector')

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

                

                # 初始化 vc (Video Clip)
                vc = None

                # --- 分支 A: 向量库 + 智能混剪 + 动效 (Vector Mode) ---
                if search_source == 'vector':
                    effect_config = scene.get('visual_effect', {})
                    candidate_pool = []
                    
                    # [关键修改] Step 1: 检查是否已有指定的本地视频 (用户已选 或 预览已锁定)
                    v_info = scene.get('video_info', {})
                    specific_path = None
                    
                    if isinstance(v_info, dict) and v_info.get('type') == 'local':
                        # 优先取 download_url (后端存储的绝对路径)，其次取 src
                        p = v_info.get('download_url') or v_info.get('src')
                        if p and os.path.exists(p):
                            specific_path = p
                    
                    if specific_path:
                        print(f"   🔒 [Locked] Using specific clip: {os.path.basename(specific_path)}")
                        # 如果锁定了视频，候选池里只放这一条
                        # get_dynamic_visuals_smart 会优先用它，如果时长不够，会根据逻辑循环它或者随机填充
                        candidate_pool = [specific_path]
                        
                    else:
                        # [原逻辑] Step 2: 没有指定视频，才进行关键词搜索
                        raw_queries = scene.get('visual_tags', [])
                        if isinstance(raw_queries, str): raw_queries = [raw_queries]
                        if not raw_queries:
                            kw = scene.get('keywords', '')
                            if kw: raw_queries = [kw]
                        
                        if raw_queries:
                            print(f"   🔍 [Auto] Searching: {raw_queries}")
                            for q in raw_queries:
                                matches = self.search_vector_match(q, top_k=3, exclude_set=global_used_paths)
                                candidate_pool.extend(matches)

                    # Step 3: 调用智能填充
                    # 注意：如果 candidate_pool 是用户锁定的 1 个视频，Smart Fill 会优先使用它。
                    # 如果这 1 个视频不够长，Smart Fill 目前的逻辑是会去随机库里拿视频补。
                    # 如果你希望“锁定了就只循环这一个，不要补其他的”，需要改 Smart Fill，但目前逻辑符合“混剪”调性。
                    vc = self.get_dynamic_visuals_smart(
                        candidate_pool, 
                        scene_total_duration, 
                        global_used_paths,
                        effect_config=effect_config
                    )

                # --- 分支 B: Pexels / 传统模式 (Legacy Mode) ---
                else:
                    print(f"   🌐 [Pexels Mode] Fetching video...")
                    # 传统的下载逻辑 (scene['video_info'] 里通常是 Pexels URL)
                    vc = self.get_video_clip_safe(video_info, scene_total_duration, log)
                    
                    if vc.duration > scene_total_duration:
                        vc = vc.subclip(0, scene_total_duration)
                    elif vc.duration < scene_total_duration:
                        vc = vc.loop(duration=scene_total_duration)
                    
                    vc = vc.set_duration(scene_total_duration)
                    

                
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