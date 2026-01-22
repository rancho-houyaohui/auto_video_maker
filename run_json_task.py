import asyncio
import json
import os
from logic import VideoEngine
import config

# 配置你的 JSON 文件路径
JSON_FILE_PATH = "content.json" 
OUTPUT_VIDEO_PATH = os.path.join(config.OUTPUT_DIR, "final_output.mp4")

def convert_custom_json_to_engine_format(custom_json):
    """
    将你的 content.json 格式转换为 logic.py 能识别的 scenes 列表
    """
    engine_scenes = []
    
    meta = custom_json.get('project_meta', {})
    timeline = custom_json.get('timeline', [])
    
    for block in timeline:
        # 提取视觉关键词 (取第一个作为主搜索词)
        visual_tags = block.get('visual_search_queries', ["abstract"])
        main_tag = visual_tags[0] if visual_tags else "abstract"
        
        # 提取高亮关键词
        highlight = block.get('center_highlight', {})
        keywords = highlight.get('text', "") if highlight.get('enabled') else ""
        
        # 提取音效
        sfx = highlight.get('sfx', "")
        
        scene = {
            "text": block['sentence_text'],
            "voice": config.DEFAULT_VOICE, # 或从 meta 中读取
            "visual_tags": main_tag, # 这里的 tag 会传给向量搜索
            "video_info": {
                "type": "local", # 标记为本地，触发 search_vector_video
                "tags": main_tag 
            },
            "keywords": keywords,
            "sfx_search": sfx,
            "is_emphasis": highlight.get('enabled', False),
            "audio_padding": 0.2 # 句间停顿
        }
        engine_scenes.append(scene)
        
    return {
        "scenes": engine_scenes,
        "bgm_file": meta.get('bgm', 'default_bgm.mp3'), # 需确保文件在 assets/music
        "bgm_volume": 0.2,
        "subtitle_style": {
            "normal": {"color": "#FFFFFF", "size": 80},
            "emphasis": {"color": "#FF0000", "size": 120}
        }
    }

async def main():
    # 1. 初始化引擎
    engine = VideoEngine()
    
    # 2. 读取 JSON
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ JSON file not found: {JSON_FILE_PATH}")
        return

    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        custom_data = json.load(f)
        
    print(f"📂 Loaded project: {custom_data.get('project_meta', {}).get('title')}")
    
    # 3. 格式转换
    render_params = convert_custom_json_to_engine_format(custom_data)
    
    # 4. 启动渲染
    print("🚀 Starting render...")
    
    # 定义简单的日志回调
    async def simple_log(msg):
        print(f"   {msg}")

    success = await engine.render_project(render_params, OUTPUT_VIDEO_PATH, log_callback=simple_log)
    
    if success:
        print(f"✅ Render finished! Output: {OUTPUT_VIDEO_PATH}")
    else:
        print("❌ Render failed.")

if __name__ == "__main__":
    asyncio.run(main())