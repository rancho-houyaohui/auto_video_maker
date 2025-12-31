import os
import asyncio
from playwright.async_api import async_playwright

COOKIE_DIR = "cookies"
if not os.path.exists(COOKIE_DIR):
    os.makedirs(COOKIE_DIR)

class CookieManager:
    def __init__(self):
        self.cookie_dir = COOKIE_DIR

    def get_cookie_path(self, platform):
        return os.path.join(self.cookie_dir, f"{platform}.json")

    def has_cookie(self, platform):
        return os.path.exists(self.get_cookie_path(platform))

    async def login_and_save_cookie(self, platform):
        """
        启动有头浏览器，让用户扫码，关闭窗口后保存 Cookie
        """
        urls = {
            "douyin": "https://creator.douyin.com/",
            "bilibili": "https://member.bilibili.com/platform/upload/video/frame",
            "kuaishou": "https://cp.kuaishou.com/profile",
            "xiaohongshu": "https://creator.xiaohongshu.com/publish/publish",
            "video_number": "https://channels.weixin.qq.com/platform", # 视频号
        }
        
        target_url = urls.get(platform)
        if not target_url:
            raise ValueError(f"Unknown platform: {platform}")

        print(f"🕵️ 正在启动浏览器进行登录: {platform}...")
        
        async with async_playwright() as p:
            # 启动有头模式 (headless=False)
            browser = await p.chromium.launch(headless=False, args=['--start-maximized'])
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = await context.new_page()
            
            await page.goto(target_url)
            
            print("⏳ 请在弹出的浏览器中扫码登录...")
            print("✅ 登录成功后，请直接【关闭浏览器窗口】即可自动保存 Cookie。")
            
            # 等待浏览器关闭 (这就是扫码的时间)
            try:
                await page.wait_for_event("close", timeout=0) # 无限等待直到用户关闭
            except:
                pass # 忽略关闭时的报错

            # 保存 Cookie
            cookie_path = self.get_cookie_path(platform)
            await context.storage_state(path=cookie_path)
            print(f"💾 Cookie 已保存至: {cookie_path}")
            return True