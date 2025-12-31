import os
import asyncio
from playwright.async_api import async_playwright

async def publish_douyin(cookie_path, video_path, title, tags=None):
    if not os.path.exists(cookie_path):
        raise Exception("未登录，请先扫码")

    async with async_playwright() as p:
        print("🚀 启动后台浏览器发布抖音...")
        # 启动无头模式
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=cookie_path)
        page = await context.new_page()
        
        try:
            # 1. 访问发布页
            await page.goto("https://creator.douyin.com/creator-micro/content/upload")
            await page.wait_for_load_state('networkidle')

            # 2. 上传视频
            # 监听文件选择器
            async with page.expect_file_chooser() as fc_info:
                # 点击上传区域 (定位可能随抖音更新而变化，需参考开源项目最新选择器)
                upload_btn = page.locator('label:has-text("点击上传")').first
                if not await upload_btn.is_visible():
                    # 备选选择器
                    upload_btn = page.locator('.upload-btn-input').first
                await upload_btn.click()
            
            file_chooser = await fc_info.value
            await file_chooser.set_files(video_path)
            print("⬆️ 视频正在上传...")
            
            # 3. 等待上传完成
            # 等待“上传成功”字样或进度条消失
            # 这是一个简单的等待逻辑，实际可能需要更复杂的判断
            await page.wait_for_selector('div:has-text("上传成功")', timeout=120000)
            print("✅ 上传成功，正在填写信息...")

            # 4. 填写标题 (包含标签)
            description = title + " " + " ".join([f"#{t}" for t in (tags or [])])
            
            # 抖音的输入框通常是一个 editor
            editor = page.locator('.zone-container').first
            await editor.click()
            await editor.fill(description)
            
            # 5. 点击发布
            publish_btn = page.locator('button:has-text("发布")').last
            await publish_btn.click()
            
            # 等待跳转或提示
            await asyncio.sleep(5)
            print("🎉 抖音发布指令已执行")
            return True

        except Exception as e:
            print(f"❌ 发布失败: {e}")
            # 截图留证
            await page.screenshot(path="error_douyin.png")
            raise e
        finally:
            await browser.close()