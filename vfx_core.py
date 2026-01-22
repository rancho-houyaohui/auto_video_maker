import numpy as np
import cv2
from moviepy.editor import *
from moviepy.video.fx.all import colorx, blackwhite

class VisualEffects:
    """
    [Plan B: OpenCV 像素级内核]
    彻底放弃 MoviePy 的动态属性 (lambda)，改用 OpenCV 逐帧处理。
    这能保证 clip.w 和 clip.h 永远是整数，彻底根除 'function + int' 报错。
    """

    @staticmethod
    def _zoom_frame(img, t, zoom_speed=0.04):
        """
        OpenCV 实现的中心放大算法
        """
        h, w = img.shape[:2]
        
        # 计算当前的缩放比例 (随时间 t 线性增加)
        scale = 1 + zoom_speed * t
        
        # 计算基于中心的裁剪区域
        # 目标是在原图中抠出一个 "越来越小" 的区域，然后放大回原尺寸 -> 看起来就是 Zoom In
        center_x, center_y = w / 2, h / 2
        
        # 裁剪框的宽和高 (随 scale 变大，裁剪框变小)
        crop_w = w / scale
        crop_h = h / scale
        
        x1 = int(center_x - crop_w / 2)
        y1 = int(center_y - crop_h / 2)
        x2 = int(center_x + crop_w / 2)
        y2 = int(center_y + crop_h / 2)
        
        # 边界保护
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 裁剪
        cropped = img[y1:y2, x1:x2]
        
        # 重新拉伸回原尺寸 (使用线性插值保证速度，或者 CUBIC 保证画质)
        # 这里的 dsize 是 (width, height)
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized

    @staticmethod
    def _pan_frame(img, t, speed=30, direction='right'):
        """
        OpenCV 实现的平移算法 (Pan)
        原理：在大图上裁切，或者简单的像素位移 (这里为了简单，做像素位移+边缘拉伸)
        注意：为了不产生黑边，Pan 通常需要原素材本身比 1080p 大。
        如果素材本身就是 1080p，强行 Pan 会出现黑边。
        为了安全，我们这里实现一个 "Zoom-Pan"：先放大一点，再移动。
        """
        h, w = img.shape[:2]
        
        # 先放大 1.1 倍，腾出移动空间
        scale = 1.1
        large_w = int(w * scale)
        large_h = int(h * scale)
        img_large = cv2.resize(img, (large_w, large_h), interpolation=cv2.INTER_LINEAR)
        
        # 基础裁剪点 (左上角)
        base_x = (large_w - w) // 2
        base_y = (large_h - h) // 2
        
        # 随时间移动偏移量
        offset = int(speed * t)
        
        if direction == 'right':
            # 向右看 -> 裁剪框向左移
            curr_x = max(0, base_x - offset)
            crop_img = img_large[base_y:base_y+h, curr_x:curr_x+w]
        else:
            # 默认 Zoom Pan
            return VisualEffects._zoom_frame(img, t, 0.05)

        # 确保尺寸对其 (防止越界导致的 1px 误差)
        if crop_img.shape[0] != h or crop_img.shape[1] != w:
            crop_img = cv2.resize(crop_img, (w, h))
            
        return crop_img

    @staticmethod
    def apply_camera_movement(clip, move_type):
        """
        应用运镜
        """
        # 1. 确保输入是标准的 1920x1080 (静态调整)
        # 这一步是为了让 OpenCV 处理时数据统一
        target_w, target_h = 1920, 1080
        
        if clip.w != target_w or clip.h != target_h:
            clip = clip.resize(height=target_h)
            if clip.w > target_w:
                clip = clip.crop(x_center=clip.w/2, width=target_w, height=target_h)
            elif clip.w < target_w:
                clip = clip.resize(width=target_w).crop(y_center=clip.h/2, width=target_w, height=target_h)

        # 2. 使用 fl (Frame Filter) 逐帧处理
        # 这种方式不会改变 clip 的元数据属性 (w, h)，所以绝对不会报 unsupported operand
        
        if move_type == "Zoom_In_Fast":
            # 这里的 t 是当前帧相对于 clip 开头的时间
            return clip.fl(lambda gf, t: VisualEffects._zoom_frame(gf(t), t, zoom_speed=0.06))
            
        elif move_type == "Zoom_In_Slow" or move_type == "Static":
            return clip.fl(lambda gf, t: VisualEffects._zoom_frame(gf(t), t, zoom_speed=0.03))

        elif move_type == "Pan_Right":
            return clip.fl(lambda gf, t: VisualEffects._pan_frame(gf(t), t, speed=40, direction='right'))

        elif move_type == "Shake":
            # 简单的 Zoom 代替 Shake (Shake 很难在无黑边的情况下做完美)
            # 或者做一个快速的 Zoom In/Out
            return clip.fl(lambda gf, t: VisualEffects._zoom_frame(gf(t), t, zoom_speed=0.1))

        elif move_type == "Slow_Mo":
            # 慢动作 (MoviePy 原生支持，修改速度是安全的)
            return clip.speedx(0.8)

        # 默认：缓慢推镜
        return clip.fl(lambda gf, t: VisualEffects._zoom_frame(gf(t), t, zoom_speed=0.03))

    @staticmethod
    def apply_filter(clip, filter_type):
        # 滤镜部分保持不变，这些是安全的
        if filter_type == "Black_White":
            return clip.fx(blackwhite)
        elif filter_type == "High_Contrast":
            return clip.fx(colorx, 0.8).fx(vfx.lum_contrast, lum=0, contrast=1.2)
        elif filter_type == "Warm_Light":
            orange = ColorClip(size=clip.size, color=(255, 180, 50)).set_opacity(0.15).set_duration(clip.duration)
            return CompositeVideoClip([clip, orange])
        elif filter_type == "Red_Tint":
            red = ColorClip(size=clip.size, color=(255, 0, 0)).set_opacity(0.3).set_duration(clip.duration)
            return CompositeVideoClip([clip, red])
        elif filter_type == "Cyberpunk":
            blue = ColorClip(size=clip.size, color=(0, 255, 255)).set_opacity(0.2).set_duration(clip.duration)
            return CompositeVideoClip([clip.fx(vfx.lum_contrast, contrast=1.3), blue])
        return clip