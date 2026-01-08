# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import copy_metadata
import sys
import os

# --- 资源配置 ---
datas = [
    ('build_assets', 'assets'),
    ('templates', 'templates')
]

# 复制元数据
datas += copy_metadata('imageio')
datas += copy_metadata('moviepy')
datas += copy_metadata('tqdm')
datas += copy_metadata('requests')
datas += copy_metadata('gradio_client')
datas += copy_metadata('huggingface_hub')
datas += copy_metadata('ollama')
datas += copy_metadata('edge_tts')

# --- 分析配置 ---
a = Analysis(
    # 【核心修复】显式列出所有 Python 源代码文件
    # 这样 PyInstaller 就绝对不会漏掉它们了
    [
        'desktop_app.py',
        'server.py',
        'logic.py',
        'config.py',
        'project_manager.py'
    ],
    pathex=['.'], # 显式指定当前目录为搜索路径
    binaries=[('bin/ffmpeg', 'bin')], 
    datas=datas,
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'engineio.async_drivers.threading',
        'moviepy.audio.fx.all',
        'moviepy.video.fx.all',
        'PIL',
        'PIL.Image',
        'numpy',
        'certifi',
        'imageio',
        'imageio.plugins',
        'imageio.plugins.ffmpeg'
    ],
    excludes=[
        'torch', 'torchaudio', 'torchvision', 'scipy', 
        'matplotlib', 'pandas', 'tkinter', 'PyQt5', 'PySide2'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

# --- EXE 配置 ---
exe = EXE(
    pyz,
    a.scripts,
    [], 
    exclude_binaries=True,
    name='AI_Video_Station',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=True, 
    target_arch=None,
    codesign_identity=None,
    # 添加Mac系统权限配置
    entitlements_file='entitlements.plist' if os.path.exists('entitlements.plist') else None,
    icon='build_assets/icon.icns' if os.path.exists('build_assets/icon.icns') else None
)

# --- COLLECT 配置 ---
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AI_Video_Station',
)

# --- BUNDLE 配置 ---
app = BUNDLE(
    coll,
    name='AI_Video_Station.app',
    icon='build_assets/icon.icns' if os.path.exists('build_assets/icon.icns') else None,
    bundle_identifier='com.rancho.aivideostation',
    info_plist={
        'CFBundleIconFile': 'icon.icns'
    },
)