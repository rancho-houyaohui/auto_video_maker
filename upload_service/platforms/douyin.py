import os
import asyncio
from playwright.async_api import async_playwright

async def publish_douyin(cookie_path, video_path, title, tags=None):
    if not os.path.exists(cookie_path):
        raise Exception("Cookie文件不存在，请先登录")
    
    if not os.path.exists(video_path):
        raise Exception(f"视频文件不存在: {video_path}")

    print(f"🚀 [Douyin] 开始发布: {title}")
    
    async with async_playwright() as p:
        # 1. 启动 (Headless=True 后台运行)
        # 注意：如果发布失败，可以临时改为 False 查看浏览器发生了什么
        browser = await p.chromium.launch(headless=True) 
        context = await browser.new_context(storage_state=cookie_path)
        page = await context.new_page()
        
        try:
            # 2. 进入发布页
            await page.goto("https://creator.douyin.com/creator-micro/content/upload")
            # --- [新增] 核心检查：是否跳到了登录页，或者没有上传按钮 ---
            try:
                # 等待“点击上传”或者“登录”相关元素
                # 如果跳到了 login 页面，或者 header 里显示“登录”，说明 cookie 失效
                # 这里简单判断：如果 10秒内没找到“上传”按钮，大概率是失效了
                await page.wait_for_selector('text=点击上传', timeout=10000)
            except:
                # 再次确认是否在登录页
                if "login" in page.url:
                    raise Exception("AUTH_EXPIRED")
                
                # 尝试截图分析
                await page.screenshot(path="debug_auth_check.png")
                raise Exception("AUTH_EXPIRED") # 抛出特定异常字符串

            # 3. 上传视频
            async with page.expect_file_chooser() as fc_info:
                # 寻找上传区域，通常是包含 input[type=file] 的区域
                upload_trigger = page.locator('label:has-text("点击上传"), .upload-btn-input').first
                await upload_trigger.click()
            
            file_chooser = await fc_info.value
            await file_chooser.set_files(video_path)
            print("⬆️ 正在上传视频...")

            # 4. 等待上传完毕 (检测“重新上传”按钮出现，或者进度条消失)
            # 这里的 timeout 设长一点，取决于网速
            await page.wait_for_selector('div:has-text("上传成功")', timeout=180000)
            print("✅ 视频上传完毕")

            # 5. 填写标题 (输入框通常是一个 contenteditable 的 div)
            full_title = f"{title} " + " ".join([f"#{t}" for t in (tags or [])])
            
            # 定位标题输入框 (抖音改版频繁，尝试多种定位)
            title_input = page.locator('.zone-container, .editor-kit-container').first
            await title_input.click()
            await title_input.fill(full_title)
            
            # 6. 处理其他弹窗 (可选)
            # 有时候会有“是否关联热点”之类的，尽量忽略

            # 7. 点击发布
            # 找到最显眼的“发布”按钮
            publish_btn = page.locator('button:has-text("发布")').last
            # 确保按钮是可点击状态 (不是 disabled)
            await expect(publish_btn).to_be_enabled(timeout=10000)
            await publish_btn.click()
            
            # 8. 确认发布成功 (跳转或提示)
            # 等待几秒
            await asyncio.sleep(5)
            print("🎉 发布动作已执行")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"❌ [Douyin] 异常: {error_msg}")
            
            # --- [核心修改] 捕获过期，删除 Cookie ---
            if "AUTH_EXPIRED" in error_msg or "过期" in error_msg or "登录" in error_msg:
                print(f"⚠️ 检测到 Cookie 失效，正在删除: {cookie_path}")
                if os.path.exists(cookie_path):
                    os.remove(cookie_path)
                # 抛出特定关键词，供 server.py 识别
                raise Exception("AUTH_EXPIRED")
            
            await page.screenshot(path="error_douyin.png")
            raise e
        finally:
            await browser.close()

# 辅助函数
from playwright.async_api import expect