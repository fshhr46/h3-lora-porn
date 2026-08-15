#!/usr/bin/env python3
"""
视频预处理脚本 - 从原始视频中抽帧、裁剪、分类
用于准备 LoRA 训练素材

使用方法:
    python preprocess.py --input data/raw --output data/cropped --fps 8 --resolution 512x512
"""

import argparse
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV 未安装")
    print("   安装: pip install opencv-python")
    print("   或使用纯 PIL 模式（较慢）")


# ============================================================
# 视频帧提取
# ============================================================

def extract_frames_opencv(video_path, output_dir, fps_target=8, resolution=(512, 512),
                          min_frames=5, max_frames=30):
    """使用 OpenCV 从视频中提取帧"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"  ⚠️  无法获取 FPS，跳过: {video_path}")
        cap.release()
        return 0

    # 计算采样间隔
    sample_interval = max(1, int(fps / fps_target))

    extracted = 0
    frame_idx = 0
    saved = 0

    while frame_idx < total_frames and saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # 每隔 sample_interval 帧保存一帧
        if frame_idx % sample_interval == 0:
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 裁剪到目标分辨率（保持宽高比）
            h, w = frame_rgb.shape[:2]
            target_w, target_h = resolution

            # 保持宽高比裁剪
            if w / h < target_w / target_h:
                new_w = w
                new_h = int(w * target_h / target_w)
            else:
                new_h = h
                new_w = int(h * target_w / target_h)

            # 中心裁剪
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            cropped = frame_rgb[top:top+target_h, left:left+target_w]

            # 保存为 PNG
            output_path = output_dir / f"{video_path.stem}_frame_{saved:04d}.png"
            pil_img = Image.fromarray(cropped)
            pil_img.save(str(output_path), 'PNG')
            saved += 1

        frame_idx += sample_interval

    cap.release()
    print(f"  ✅ 从 {video_path.name} 提取了 {saved} 帧 "
          f"(总共 {total_frames} 帧, FPS={fps})")
    return saved


def extract_frames_pil(video_path, output_dir, fps_target=8, resolution=(512, 512),
                       min_frames=5, max_frames=30):
    """使用 PIL 从视频中提取帧（备用方案）"""
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("  ❌ moviepy 未安装")
        return 0

    clip = VideoFileClip(str(video_path))
    duration = clip.duration
    fps = clip.fps

    if fps <= 0:
        return 0

    extracted = 0
    saved = 0

    for t in range(0, int(duration * fps_target), fps_target):
        if saved >= max_frames:
            break

        if t >= duration:
            break

        frame = clip.get_frame(t)

        # HWC -> CHW, RGB
        frame_rgb = frame.transpose(2, 0, 1)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize(resolution, Image.Resampling.LANCZOS)

        output_path = output_dir / f"{video_path.stem}_frame_{saved:04d}.png"
        pil_img.save(str(output_path), 'PNG')
        saved += 1

    clip.close()
    print(f"  ✅ 从 {video_path.name} 提取了 {saved} 帧")
    return saved


# ============================================================
# 质量筛选
# ============================================================

def calculate_brightness(image_array):
    """计算图像亮度"""
    return np.mean(image_array)


def calculate_sharpness(image_array):
    """计算图像清晰度（基于方差）"""
    gray = np.mean(image_array, axis=2) if image_array.ndim == 3 else image_array
    return np.var(gray)


def filter_frames(input_dir, output_dir, min_brightness=30, max_brightness=240,
                  min_sharpness=500):
    """筛选高质量帧"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    kept = 0
    removed = 0

    for img_path in sorted(input_path.glob('*.png')):
        img = np.array(Image.open(img_path))

        brightness = calculate_brightness(img)
        sharpness = calculate_sharpness(img)

        # 过滤过暗/过亮/模糊的帧
        if (brightness < min_brightness or brightness > max_brightness or
                sharpness < min_sharpness):
            removed += 1
            continue

        # 保留质量好的帧
        img_path.rename(output_path / img_path.name)
        kept += 1

    print(f"\n📊 质量筛选结果:")
    print(f"   保留: {kept} 帧")
    print(f"   移除: {removed} 帧")
    return kept, removed


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='视频预处理脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='原始视频目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录（裁剪后的帧）')
    parser.add_argument('--filtered', type=str, default=None,
                        help='筛选后的目录（可选）')
    parser.add_argument('--fps', type=int, default=8,
                        help='目标帧率')
    parser.add_argument('--resolution', type=str, default='512x512',
                        help='输出分辨率 (WxH)')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='每个视频最大提取帧数')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = tuple(map(int, args.resolution.split('x')))
    print(f"🎬 视频预处理")
    print(f"   输入: {input_dir}")
    print(f"   输出: {output_dir}")
    print(f"   分辨率: {resolution}")
    print(f"   目标帧率: {args.fps} FPS")
    print("=" * 60)

    # 找到所有视频文件
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')
    video_files = [f for ext in video_extensions
                   for f in input_dir.glob(f'*{ext}')]

    if not video_files:
        print(f"❌ 在 {input_dir} 中未找到任何视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件\n")

    total_extracted = 0
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")

        if HAS_CV2:
            extracted = extract_frames_opencv(
                video_path, output_dir,
                fps_target=args.fps,
                resolution=resolution,
                max_frames=args.max_frames
            )
        else:
            extracted = extract_frames_pil(
                video_path, output_dir,
                fps_target=args.fps,
                resolution=resolution,
                max_frames=args.max_frames
            )

        total_extracted += extracted
        print()

    print("=" * 60)
    print(f"✅ 共提取 {total_extracted} 帧到 {output_dir}")

    # 质量筛选
    if args.filtered:
        print(f"\n🔍 开始质量筛选...")
        filter_frames(output_dir, args.filtered)
        print(f"\n筛选后的素材在: {args.filtered}")

    print("\n下一步: 人工检查筛选后的素材")
    print("   将不满意的帧从 data/filtered/ 中移除")
    print("\n素材准备完成后，运行训练:")
    print("   bash train.sh")


if __name__ == '__main__':
    main()
