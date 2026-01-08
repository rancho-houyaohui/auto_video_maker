import sys
import os
import traceback
import threading
import time


# --- [新增] SSL 证书修复 (解决打包后无法联网/HTTPS报错) ---
# 必须在所有网络库导入之前执行
if getattr(sys, 'frozen', False):
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    print(f"🔒 SSL Cert set to: {os.environ['SSL_CERT_FILE']}")

# --- 1. 日志重定向设置 ---
# 使用config.py中已定义的目录，避免重复创建
log_dir = os.path.join(os.path.expanduser("~"), "Documents", "AI_Video_Output")

log_file = os.path.join(log_dir, "app_debug.log")

class LoggerWriter:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.log_f = open(log_file, "a", encoding="utf-8", buffering=1) 

    def write(self, message):
        try:
            self.log_f.write(message)
            if self.original_stream:
                self.original_stream.write(message)
                self.original_stream.flush()
        except: pass 

    def flush(self):
        try:
            self.log_f.flush()
            if self.original_stream: self.original_stream.flush()
        except: pass

    def isatty(self):
        return False

# 立即重定向
sys.stdout = LoggerWriter(sys.stdout)
sys.stderr = LoggerWriter(sys.stderr)

print(f"🚀 Booting at {time.strftime('%Y-%m-%d %H:%M:%S')}...")

# --- 2. 导入业务模块 ---
try:
    import uvicorn
    import webview
    import config
    from server import app
    print("✅ All modules imported successfully.")
except Exception as e:
    print("❌ CRITICAL IMPORT ERROR:")
    print(traceback.format_exc())
    sys.exit(1)

# --- 3. 启动逻辑 ---
def start_server():
    try:
        log_config = uvicorn.config.LOGGING_CONFIG
        log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
        log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
        log_config["formatters"]["default"]["use_colors"] = False
        log_config["formatters"]["access"]["use_colors"] = False

        # 绑定 localhost
        uvicorn.run(app, host="127.0.0.1", port=18888, log_level="info", log_config=log_config)
        
    except Exception as e:
        print(f"❌ Server Start Error: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    if sys.platform == 'darwin':
        sys.argv = [arg for arg in sys.argv if not arg.startswith('-psn')]

    try:
        print("🔧 Initializing application...")
        
        print("🔌 Starting Server Thread...")
        t = threading.Thread(target=start_server)
        t.daemon = True
        t.start()

        time.sleep(1) # 等待 Server 启动

        # 添加窗口关闭事件处理
        def on_window_closed():
            print("🔒 Window closed, exiting application...")
            # 确保应用程序能够正常退出，不会创建新窗口
            # 使用更强制的方式退出，确保所有进程都被终止
            import os
            import signal
            if sys.platform == 'win32':
                # Windows 平台
                os._exit(0)
            else:
                # Unix/Linux/macOS 平台
                os.kill(os.getpid(), signal.SIGTERM)
        
        print("🖥️ Creating WebView...")
        # 配置webview，禁用多进程模式以避免窗口持续弹出
        window = webview.create_window(
            title='AI 视频工作站', 
            url='http://127.0.0.1:18888',
            width=1280,
            height=800,
            resizable=True,
            text_select=True,
            js_api={
                'isapp': True
            }
        )
        

        webview.start(debug=False)
        
        # 窗口关闭后，确保应用程序完全退出
        print("✅ Application exited normally.")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        print(traceback.format_exc())