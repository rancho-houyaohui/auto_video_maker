import os
from playwright.async_api import async_playwright

# 确保路径指向根目录下的 cookies
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOKIE_DIR = os.path.join(BASE_DIR, "cookies")

if not os.path.exists(COOKIE_DIR):
    os.makedirs(COOKIE_DIR)

class CookieManager:
    def __init__(self):
        self.cookie_dir = COOKIE_DIR

    def get_cookie_path(self, platform):
        return os.path.join(self.cookie_dir, f"{platform}.json")

    def has_cookie(self, platform):
        path = self.get_cookie_path(platform)
        return os.path.exists(path)

    async def login_and_save_cookie(self, platform):
        urls = {
            "douyin": "https://creator.douyin.com/",
            "bilibili": "https://member.bilibili.com/platform/upload/video/frame",
            "xiaohongshu": "https://creator.xiaohongshu.com/publish/publish",
            "video_number": "https://channels.weixin.qq.com/platform",
        }
        
        target_url = urls.get(platform)
        if not target_url: return False

        print(f"🕵️ 启动浏览器登录: {platform}...")
        
        async with async_playwright() as p:
            # 启动有头浏览器
            browser = await p.chromium.launch(headless=False, args=['--start-maximized'])
            context = await browser.new_context(no_viewport=True) # 禁用视口限制
            page = await context.new_page()
            
            await page.goto(target_url)
            
            print("⏳ 请扫码登录，登录成功后请【手动关闭浏览器窗口】...")
            
            # 核心：等待浏览器被用户关闭
            try:
                # 这是一个无限等待，直到用户点 X 关闭窗口，以此作为“登录完成”的信号
                await page.wait_for_event("close", timeout=0) 
            except:
                pass 

            # 保存 Cookie
            cookie_path = self.get_cookie_path(platform)
            await context.storage_state(path=cookie_path)
            print(f"💾 Cookie 已保存: {cookie_path}")
            return True