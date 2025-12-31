from uploader.cookie_manager import CookieManager
from uploader.platforms import douyin
# from uploader.platforms import bilibili # 以后扩展

class UploaderEngine:
    def __init__(self):
        self.cookie_mgr = CookieManager()
    
    def check_login_status(self):
        """返回各平台的登录状态"""
        platforms = ["douyin", "bilibili", "video_number", "xiaohongshu", "kuaishou"]
        status = {}
        for p in platforms:
            status[p] = self.cookie_mgr.has_cookie(p)
        return status

    async def login_platform(self, platform):
        return await self.cookie_mgr.login_and_save_cookie(platform)

    async def publish_video(self, platform, video_path, title, tags=[]):
        cookie_path = self.cookie_mgr.get_cookie_path(platform)
        
        if platform == "douyin":
            return await douyin.publish_douyin(cookie_path, video_path, title, tags)
        # elif platform == "bilibili":
        #     return await bilibili.publish_bilibili(...)
        else:
            raise Exception(f"暂未集成该平台: {platform}")