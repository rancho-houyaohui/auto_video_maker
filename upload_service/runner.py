from upload_service.cookie_manager import CookieManager
from upload_service.adapter import SocialUploaderAdapter # 引入适配器
import config
import os

class UploaderEngine:
    def __init__(self):
        self.cookie_mgr = CookieManager()
        # 实例化适配器，传入 cookie 目录
        self.adapter = SocialUploaderAdapter(self.cookie_mgr.cookie_dir)
    
    def check_login_status(self):
        platforms = ["douyin", "bilibili", "video_number", "xiaohongshu", "kuaishou"]
        status = {}
        for p in platforms:
            status[p] = self.cookie_mgr.has_cookie(p)
        return status

    async def login_platform(self, platform):
        # 登录逻辑还是用我们自己写的简单扫码器，比较纯净
        # 只要保存的文件路径和 Adapter 读取的路径一致即可
        return await self.cookie_mgr.login_and_save_cookie(platform)

    async def publish_video(self, platform, video_path, title, tags=[]):
        # 核心：调用适配器进行发布
        await self.adapter.upload(
            platform=platform,
            video_path=video_path,
            title=title,
            tags=tags
        )