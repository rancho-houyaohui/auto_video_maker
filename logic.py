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
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np


# 修复 PIL
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

class VideoEngine:
    def __init__(self):
        self.ASSETS_DIR = config.ASSETS_DIR
        self.TEMP_DIR = config.TEMP_DIR
        self.FONT_PATH = config.FONT_PATH
        self.runtime_pexels_key = ""
        self.runtime_pixabay_key = ""
        
        for d in ["video", "sfx", "music", "fonts", "outputs"]: 
            os.makedirs(os.path.join(self.ASSETS_DIR, d), exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def set_api_keys(self, pexels, pixabay):
        self.runtime_pexels_key = pexels.strip()
        self.runtime_pixabay_key = pixabay.strip()

    def _get_key(self, key_type):
        if key_type == "pexels":
            return self.runtime_pexels_key if self.runtime_pexels_key else config.PEXELS_API_KEY
        elif key_type == "pixabay":
            return self.runtime_pixabay_key if self.runtime_pixabay_key else config.PIXABAY_API_KEY
        return ""

    def sanitize_filename(self, name):
        name = str(name).replace(" ", "_")
        return re.sub(r'[^\w\-_]', '', name)

    # --- [核心优化] 文本分段算法 (更连贯) ---
    def split_text_by_breath(self, text):
        return self.smart_split_text(text)

    def smart_split_text(self, text, min_chars=15): # 默认改为15字
        """
        1. 移除换行。
        2. 优先按强逻辑符号(句号/问号/感叹号)切分。
        3. 只有当缓冲区过长时，才考虑按逗号切分。
        """
        text = text.replace("\n", " ").strip()
        if not text: return []
        
        # 将所有强结束符替换为特殊标记，方便split
        # 保护中文省略号...
        text = text.replace("...", "@@ELLIPSIS@@")
        
        # 1. 强切分：按 。！？ ； 分割
        # 这里的正则意味着：遇到这些符号就强制切一刀
        sentences = re.split(r'([。！？!?;；]+)', text)
        
        final_chunks = []
        current_chunk = ""
        
        for part in sentences:
            # 还原省略号
            part = part.replace("@@ELLIPSIS@@", "...")
            
            # 如果是标点，追加到上一句并强制结束当前句
            if re.match(r'^[。！？!?;；]+$', part):
                current_chunk += part
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                current_chunk = ""
                continue
            
            # 如果是文本
            # 检查加上这段话是否太长
            # 如果 current_chunk 已经很长了，且 part 之前是逗号分隔的（隐性逻辑），尝试拆
            # 但为了简单，我们这里先合并，如果合并后太长，再内部按逗号拆
            
            temp_combined = current_chunk + part
            
            # 如果合并后长度还可以，或者 part 本身就很短，就合并
            if len(temp_combined) < min_chars * 2: # 允许稍微长一点
                current_chunk += part
            else:
                # 如果太长了，尝试在 part 内部找逗号切分
                # 这里简化处理：直接合并，利用标点逻辑
                current_chunk += part

        # 处理末尾
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())
            
        # --- 二次处理：检查有没有特别长的句子，按逗号细分 ---
        refined_chunks = []
        for chunk in final_chunks:
            if len(chunk) > 25: # 如果一句话超过25字，必须切
                # 按逗号切，但要保证切完的每段不短于 min_chars/2
                commas = re.split(r'([，,])', chunk)
                sub_buf = ""
                for frag in commas:
                    if re.match(r'[，,]', frag):
                        sub_buf += frag
                        # 只有当 sub_buf 足够长时才切，否则保留逗号继续攒
                        if len(sub_buf) > 10: 
                            refined_chunks.append(sub_buf)
                            sub_buf = ""
                    else:
                        sub_buf += frag
                if sub_buf: refined_chunks.append(sub_buf)
            else:
                refined_chunks.append(chunk)

        return [c for c in refined_chunks if c.strip()]

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
Style: Normal,Arial,{n_size},{n_color},{n_color},{n_outline},&H80000000,1,0,0,0,100,100,0,0,1,3,2,2,30,30,60,1
Style: Emphasis,Arial,{e_size},{e_color},{e_color},{e_outline},&H00000000,1,0,0,0,100,100,0,0,1,5,0,5,30,30,30,1

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

    async def run_ffmpeg_async(self, cmd, log_callback):
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        while True:
            line = await process.stderr.readline()
            if not line: break
            line_str = line.decode('utf-8', errors='ignore').strip()
            if line_str:
                if "frame=" in line_str or "size=" in line_str:
                    await log_callback(f"[FFmpeg] {line_str}")
                elif "Error" in line_str:
                    await log_callback(f"⚠️ {line_str}")
        await process.wait()
        if process.returncode != 0:
            await log_callback(f"❌ FFmpeg return code: {process.returncode}")

    # --- 资源功能 ---
    def search_local_videos(self, tag):
        video_dir = os.path.join(self.ASSETS_DIR, "video")
        if not os.path.exists(video_dir): return []
        files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov'))]
        matches = [f for f in files if tag.lower() in f.lower()]
        return matches if matches else files

    def download_video(self, query):
        key = self._get_key("pexels")
        if not key or "粘贴" in key: return None
        print(f"🌐 Downloading: {query}")
        headers = {"Authorization": key, "User-Agent": "Mozilla/5.0"}
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=1&orientation=landscape"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            if not data.get('videos'): return None
            vid_data = data['videos'][0]
            raw_tags = vid_data.get('tags', [])
            tag_slug = self.sanitize_filename(query)
            extra_tags = "_".join([self.sanitize_filename(t)[:10] for t in raw_tags[:5]])
            fname = f"pexels_{vid_data['id']}_{tag_slug}_{extra_tags}.mp4"
            if len(fname) > 200: fname = fname[:200] + ".mp4"
            video_files = vid_data['video_files']
            target = next((vf['link'] for vf in video_files if vf['width']==1920), video_files[0]['link'])
            content = requests.get(target, timeout=60).content
            path = os.path.join(self.ASSETS_DIR, "video", fname)
            with open(path, 'wb') as f: f.write(content)
            return fname
        except Exception as e:
            print(f"DL Error: {e}")
            return None

    def download_video_by_url(self, download_url, video_id, tags):
        try:
            print(f"🌐 Direct Downloading: ID {video_id}")
            tag_slug = self.sanitize_filename(tags)
            if len(tag_slug) > 50: tag_slug = tag_slug[:50]
            fname = f"pexels_{video_id}_{tag_slug}.mp4"
            path = os.path.join(self.ASSETS_DIR, "video", fname)
            if os.path.exists(path): return fname
            headers = {"User-Agent": "Mozilla/5.0"}
            content = requests.get(download_url, headers=headers, timeout=120).content
            with open(path, 'wb') as f: f.write(content)
            print(f"✅ 下载完成: {fname}")
            return fname
        except Exception as e:
            print(f"Direct DL Error: {e}")
            return None

    def search_online_videos(self, query):
        key = self._get_key("pexels")
        if not key or "粘贴" in key: return {"error": "API Key 未配置"}
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
                    "type": "online",
                    "id": v['id'],
                    "src": preview,
                    "download_url": download,
                    "tags": tags_str,
                    "name": f"Pexels ID: {v['id']}"
                })
            return results
        except Exception as e:
            return {"error": str(e)}

    # --- 音效搜索 (MyInstants) ---
    def search_online_sfx(self, query):
        print(f"🔊 Searching SFX: {query}")
        url = "https://www.myinstants.com/api/v1/instants/"
        params = {"name": query, "format": "json"}
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 200: return {"error": f"API Error: {r.status_code}"}
            data = r.json()
            results = []
            for item in data.get('results', [])[:15]:
                sound_url = item.get('sound')
                if not sound_url: continue
                name = item.get('name', 'SFX').replace("'", "").replace('"', '')
                results.append({
                    "id": str(random.randint(10000, 99999)),
                    "name": name,
                    "duration": 2,
                    "download_url": sound_url, 
                    "preview_url": sound_url 
                })
            return results
        except Exception as e:
            return {"error": str(e)}

    def download_sfx_manual(self, query, download_url=None):
        sfx_dir = os.path.join(self.ASSETS_DIR, "sfx")
        if download_url:
            try:
                base_name = f"auto_{self.sanitize_filename(query)}"
                fname = f"{base_name}.mp3"
                save_path = os.path.join(sfx_dir, fname)
                if os.path.exists(save_path):
                    fname = f"{base_name}_{random.randint(100,999)}.mp3"
                    save_path = os.path.join(sfx_dir, fname)
                print(f"🔊 下载音效 URL: {query}")
                c = requests.get(download_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30).content
                with open(save_path, 'wb') as f: f.write(c)
                return fname
            except Exception as e:
                print(f"SFX DL Error: {e}")
                return None
        else:
            return self.get_dynamic_sfx(query, sfx_dir)

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
            # --- [核心修复] 强制视频逻辑 ---
            # 1. 循环以满足时长
            if vc.duration < duration:
                vc = vc.loop(duration=duration)
            else:
                # 2. 随机截取
                max_s = max(0, vc.duration - duration - 0.1)
                s = random.uniform(0, max_s)
                vc = vc.subclip(s, s+duration)
            
            # Resize
            vc = vc.resize(height=1080)
            if vc.w > 1920: vc = vc.crop(x1=vc.w/2-960, width=1920, height=1080)
            elif vc.w < 1920: vc = vc.resize(width=1920).crop(x1=0, y1=vc.h/2-540, width=1920, height=1080)
            
            return vc
        except:
            return ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)

    def get_dynamic_sfx(self, search_term, save_dir):
        if not search_term: return None
        local_files = [f for f in os.listdir(save_dir) if not f.startswith('.')]
        for f in local_files:
            if search_term.lower() in f.lower(): return os.path.join(save_dir, f)
        return None
        
    def load_sfx_resource(self, sfx_query, sfx_dir):
        """优先文件名精确查找，其次按关键词模糊查找"""
        if not sfx_query: return None
        exact_path = os.path.join(sfx_dir, sfx_query)
        if os.path.exists(exact_path) and os.path.isfile(exact_path):
            return exact_path
        local_files = [f for f in os.listdir(sfx_dir) if not f.startswith('.')]
        for f in local_files:
            if sfx_query.lower() in f.lower():
                return os.path.join(sfx_dir, f)
        return None

    def analyze_script(self, text):
        print(f"🤖 Calling LLM: {config.MODEL_NAME}...")
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
            r = ollama.chat(model=config.MODEL_NAME, messages=[{'role':'user','content':prompt}])
            c = r['message']['content']
            s = c.find('[')
            e = c.rfind(']')+1
            scenes = json.loads(c[s:e])
            
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
        async def log(msg):
            print(msg)
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

                # 【修复】使用智能分段
                sub_chunks = self.smart_split_text(raw_text_clean, min_chars=15)
                
                scene_audio_clips = []
                scene_total_duration = 0.0
                
                for sub_idx, chunk in enumerate(sub_chunks):
                    tts_text = chunk.strip()
                    if not tts_text or re.match(r'^[，。！?？,!.、\s]+$', tts_text): continue
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
                
                # --- 【修复】音效加载 ---
                if sfx:
                    # 使用 load_sfx_resource 查找文件
                    sp = self.load_sfx_resource(sfx, os.path.join(self.ASSETS_DIR, "sfx"))
                    if sp: 
                        await log(f"   🔊 添加音效: {os.path.basename(sp)}")
                        try: final_audio.append(AudioFileClip(sp).volumex(0.6).set_start(0))
                        except Exception as e: await log(f"   ⚠️ 音效加载失败: {e}")
                
                vc = self.get_video_clip_safe(video_info, scene_total_duration, log)
                
                # --- 【核心修复】强制视频时长 ---
                vc = vc.set_duration(scene_total_duration) # 锁死时长
                vc = vc.set_audio(CompositeAudioClip(final_audio))
                
                scene_out = os.path.join(self.TEMP_DIR, f"scene_{idx}.mov") # 使用 .mov + pcm_s16le
                vc.write_videofile(
                    scene_out, 
                    fps=30, 
                    preset="ultrafast", 
                    logger=None, 
                    codec="libx264", 
                    audio_codec="pcm_s16le", 
                    temp_audiofile=f"{self.TEMP_DIR}/temp_{idx}.wav", 
                    remove_temp=True
                )
                
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
            await self.run_ffmpeg_async(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", temp_concat], log_callback)
            
            await log("🎵 混合BGM...")
            final_clip = VideoFileClip(temp_concat)
            if bgm_file:
                bp = os.path.join(self.ASSETS_DIR, "music", bgm_file)
                if os.path.exists(bp):
                    try:
                        bgm = AudioFileClip(bp).volumex(bgm_vol)
                        from moviepy.audio.fx.all import audio_loop
                        bgm = audio_loop(bgm, duration=final_clip.duration)
                        if final_clip.audio:
                            new_audio = CompositeAudioClip([final_clip.audio, bgm])
                        else:
                            new_audio = bgm
                        final_clip = final_clip.set_audio(new_audio)
                    except: pass
            
            temp_video_bgm = os.path.join(self.TEMP_DIR, "temp_with_bgm.mov")
            final_clip.write_videofile(temp_video_bgm, fps=30, codec="libx264", audio_codec="pcm_s16le", temp_audiofile="temp_final.wav", remove_temp=True, logger=None)
            final_clip.close()

            await log("📝 压制字幕与最终输出...")
            ass_str = self.generate_ass_header(sub_style)
            for l in subtitles_events: ass_str += l + "\n"
            ass_p = os.path.join(self.TEMP_DIR, "s.ass")
            with open(ass_p, "w", encoding="utf-8") as f: f.write(ass_str)
            
            fdir = os.path.abspath(os.path.join(self.ASSETS_DIR, "fonts")).replace("\\", "/")
            font_p = os.path.join(fdir, "font.ttf")
            vf = f"ass={ass_p}:fontsdir={fdir}" if os.path.exists(font_p) else f"ass={ass_p}"
            
            await self.run_ffmpeg_async([
                "ffmpeg", "-y", "-i", temp_video_bgm, 
                "-vf", vf, 
                "-c:v", "libx264", "-preset", "fast", "-crf", "23", 
                "-c:a", "aac", "-b:a", "192k", 
                output_file
            ], log_callback)
            
            os.remove(temp_video_bgm); os.remove(temp_concat); os.remove(list_path); os.remove(ass_p)
            shutil.rmtree(self.TEMP_DIR); os.makedirs(self.TEMP_DIR, exist_ok=True)
            
            final_filename = os.path.basename(output_file)
            final_url = f"/outputs/{final_filename}"
            await log(f"✅ 处理完成@@@{final_url}")
            return True

        except Exception as e:
            await log(f"❌ Error: {traceback.format_exc()}")
            return False