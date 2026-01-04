import sys
import os
import asyncio
from pathlib import Path

# --- 1. 定位开源项目路径 ---
# 获取当前文件 (adapter.py) 所在目录的上一级 (auto_video_maker)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 指向 third_party/social_auto_upload
third_party_path = os.path.join(project_root, "third_party", "social_auto_upload")

# --- 2. 检查路径是否存在 ---
if not os.path.exists(os.path.join(third_party_path, "uploader")):
    print(f"❌ 严重错误：在 {third_party_path} 下找不到 uploader 文件夹。")
    print("请检查：1. 是否解压了开源项目？ 2. 文件夹名是否正确？")
else:
    # --- 3. 将开源项目路径插入到 sys.path 的最前面 ---
    # 这样 import uploader 就会优先去开源项目里找，而不是找本地的
    if third_party_path not in sys.path:
        sys.path.insert(0, third_party_path)

# --- 4. 尝试导入开源模块 ---
try:
    # 这里的 uploader 指的是 third_party 里的
    from uploader.douyin_uploader.main import DouYinVideo
    from uploader.bilibili_uploader.main import BilibiliVideo
    from uploader.xhs_uploader.main import XHSVideo
    from uploader.tencent_uploader.main import WeChatVideo
    print("✅ 第三方上传库加载成功")
except ImportError as e:
    print(f"⚠️ 导入第三方库失败: {e}")
    print("请尝试在 third_party/social_auto_upload 目录下运行: pip install -r requirements.txt")

class SocialUploaderAdapter:
    def __init__(self, cookie_root):
        self.cookie_root = cookie_root

    def _get_cookie_path(self, platform):
        # 映射平台名称到 cookie 文件名
        # 开源项目通常直接读取 json 文件路径
        return os.path.join(self.cookie_root, f"{platform}.json")

    async def upload(self, platform, video_path, title, tags, thumbnail_path=None):
        cookie_file = self._get_cookie_path(platform)
        if not os.path.exists(cookie_file):
            raise Exception(f"Cookie文件不存在: {cookie_file}，请先去【发布中心】扫码登录。")

        # 构造描述
        publish_title = title
        # 抖音/小红书等通常把 tag 写在描述里
        publish_desc = f"{title}\n" + " ".join([f"#{t}" for t in tags])
        
        # 转换为 Path 对象 (开源库要求)
        video_file = Path(video_path)
        
        print(f"🚀 [Adapter] 调用开源库发布到 {platform}...")

        # --- 分平台调用 ---
        if platform == 'douyin':
            # DouYinVideo(title, file_path, tags, publish_date, account_file, thumbnail_path=None, ...)
            app = DouYinVideo(
                title=publish_title,
                file_path=video_file,
                tags=tags,
                publish_date=publish_date,
                account_file=cookie_file
            )
            await app.main()
            
        elif platform == 'bilibili':
            app = BilibiliVideo(
                title=publish_title,
                file_path=video_file,
                tags=tags,
                description=publish_desc,
                account_file=cookie_file
            )
            await app.main()
            
        elif platform == 'xiaohongshu':
            app = XHSVideo(
                title=publish_title,
                file_path=video_file,
                tags=tags,
                publish_date=publish_date,
                account_file=cookie_file
            )
            await app.main()
            
        elif platform == 'video_number': # 视频号
            app = WeChatVideo(
                title=publish_title,
                file_path=video_file,
                tags=tags,
                publish_date=publish_date,
                account_file=cookie_file
            )
            await app.main()
            
        else:
            raise ValueError(f"暂不支持平台: {platform}")
            
        print(f"✅ [Adapter] {platform} 发布流程结束")