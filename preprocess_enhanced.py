#!/usr/bin/env python3
"""
增强版预处理脚本 - 支持静态图片 + 视频序列训练

功能:
1. 从视频抽帧 → 静态图片（用于普通 LoRA）
2. 提取有序帧序列 → 时序 LoRA 训练
3. 提取相邻帧对 → 帧差 LoRA 训练
4. 质量筛选（亮度、清晰度）

使用方法:
    # 普通静态图片训练
    python preprocess_enhanced.py --mode images --input data/raw --output data/cropped

    # 时序序列训练
    python preprocess_enhanced.py --mode temporal --input data/raw --output data/temporal --seq_length 8

    # 帧差训练
    python preprocess_enhanced.py --mode framediff --input data/raw --output data/framediff
"""

import argparse
import os
import sys
import time
import json
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


# ============================================================
# 通用工具
# ============================================================

def calculate_brightness(image_array):
    """计算图像亮度"""
    return np.mean(image_array)


def calculate_sharpness(image_array):
    """计算图像清晰度（基于拉普拉斯方差）"""
    gray = np.mean(image_array, axis=2) if image_array.ndim == 3 else image_array
    gray = gray.astype(np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F) if HAS_CV2 else np.var(gray)
    return np.var(laplacian) if HAS_CV2 else np.var(gray)


def filter_by_quality(frames, min_brightness=30, max_brightness=240,
                      min_sharpness=500):
    """按质量筛选帧"""
    kept = []
    removed = 0

    for i, frame in enumerate(frames):
        brightness = calculate_brightness(frame)
        sharpness = calculate_sharpness(frame)

        if (brightness < min_brightness or brightness > max_brightness or
                sharpness < min_sharpness):
            removed += 1
            continue

        kept.append((i, frame))

    return kept, removed


# ============================================================
# 模式 1: 静态图片提取
# ============================================================

def extract_images(video_dir, output_dir, fps_target=8, resolution=(512, 512),
                   max_frames=30, min_brightness=30, max_brightness=240,
                   min_sharpness=500):
    """从视频中提取静态图片用于普通 LoRA 训练"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif')
    video_files = [f for ext in video_extensions for f in video_dir.glob(f'*{ext}')]

    if not video_files:
        print(f"❌ 在 {video_dir} 中未找到任何视频文件")
        return 0

    print(f"📹 找到 {len(video_files)} 个视频文件")
    print(f"🎯 提取模式: 静态图片 (普通 LoRA)")
    print("=" * 60)

    total_extracted = 0
    total_filtered = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  ❌ 无法打开")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            cap.release()
            continue

        sample_interval = max(1, int(fps / fps_target))
        extracted = 0
        frame_idx = 0

        while frame_idx < total_frames and extracted < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 裁剪
                h, w = frame_rgb.shape[:2]
                target_w, target_h = resolution
                if w / h < target_w / target_h:
                    new_w, new_h = w, int(w * target_h / target_w)
                else:
                    new_h, new_w = h, int(h * target_w / target_h)

                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                cropped = frame_rgb[top:top+target_h, left:left+target_w]

                # 保存为 PNG
                output_path = output_dir / f"{video_path.stem}_frame_{extracted:04d}.png"
                pil_img = Image.fromarray(cropped)
                pil_img.save(str(output_path), 'PNG')
                extracted += 1

            frame_idx += sample_interval

        cap.release()
        print(f"  ✅ 提取了 {extracted} 帧")
        total_extracted += extracted

    print("\n" + "=" * 60)
    print(f"✅ 共提取 {total_extracted} 帧到 {output_dir}")
    print("\n📊 使用这些图片训练普通 LoRA:")
    print("   python train_lora.py --data_dir data/cropped")
    return total_extracted


# ============================================================
# 模式 2: 时序序列提取
# ============================================================

def extract_temporal(video_dir, output_dir, seq_length=8, fps_target=8,
                     resolution=(512, 512)):
    """提取有序帧序列用于时序 LoRA 训练"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for ext in video_extensions for f in video_dir.glob(f'*{ext}')]

    if not video_files:
        print(f"❌ 未找到视频文件")
        return 0

    print(f"📹 找到 {len(video_files)} 个视频文件")
    print(f"🎯 提取模式: 时序序列 (Temporal LoRA)")
    print(f"   序列长度: {seq_length} 帧")
    print("=" * 60)

    total_sequences = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            cap.release()
            continue

        sample_interval = max(1, int(fps / fps_target))

        # 提取所有采样帧
        all_frames = []
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, resolution)
            all_frames.append(frame_rgb)
            frame_idx += sample_interval

        cap.release()

        # 切分成序列
        seq_count = max(0, len(all_frames) - seq_length + 1)
        if seq_count == 0:
            print(f"  ⚠️  帧数不够 ({len(all_frames)} < {seq_length})")
            continue

        # 保存每个序列为一个 numpy 文件（保持顺序）
        seq_dir = output_dir / video_path.stem
        seq_dir.mkdir(exist_ok=True)

        sequences_saved = min(seq_count, 10)  # 每个视频最多保存 10 个序列
        for s in range(sequences_saved):
            start = np.random.randint(0, max(1, len(all_frames) - seq_length))
            seq_frames = all_frames[start:start + seq_length]

            # 保存为 numpy 数组
            np.save(str(seq_dir / f"seq_{s:04d}.npy"), np.array(seq_frames))

            # 同时保存第一帧作为缩略图
            img = Image.fromarray(seq_frames[0])
            img.save(str(seq_dir / f"seq_{s:04d}_preview.png"))

        print(f"  ✅ 提取了 {sequences_saved} 个序列 "
              f"(共 {seq_count} 个可用)")
        total_sequences += sequences_saved

    print("\n" + "=" * 60)
    print(f"✅ 共提取 {total_sequences} 个序列到 {output_dir}")
    print("\n📊 使用这些序列训练时序 LoRA:")
    print("   python train_video_lora.py --mode temporal --video_dir data/temporal")
    return total_sequences


# ============================================================
# 模式 3: 帧差对提取
# ============================================================

def extract_frame_diff(video_dir, output_dir, resolution=(256, 256)):
    """提取相邻帧对用于帧差 LoRA 训练"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for ext in video_extensions for f in video_dir.glob(f'*{ext}')]

    if not video_files:
        print(f"❌ 未找到视频文件")
        return 0

    print(f"📹 找到 {len(video_files)} 个视频文件")
    print(f"🎯 提取模式: 帧差对 (Frame Diff LoRA)")
    print("=" * 60)

    total_pairs = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 2:
            cap.release()
            continue

        pairs_saved = 0
        max_pairs = 50  # 每个视频最多提取 50 对

        for frame_idx in range(0, total_frames - 1, 3):  # 每隔 3 帧取一对
            if pairs_saved >= max_pairs:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret1, frame1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 1)
            ret2, frame2 = cap.read()

            if not ret1 or not ret2:
                continue

            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            frame1_rgb = cv2.resize(frame1_rgb, resolution)
            frame2_rgb = cv2.resize(frame2_rgb, resolution)

            # 保存帧对
            out_dir = output_dir / video_path.stem
            out_dir.mkdir(exist_ok=True)

            # 保存 frame A
            img_a = Image.fromarray(frame1_rgb)
            img_a.save(str(out_dir / f"frame_a_{pairs_saved:04d}.png"))

            # 保存 frame B
            img_b = Image.fromarray(frame2_rgb)
            img_b.save(str(out_dir / f"frame_b_{pairs_saved:04d}.png"))

            # 保存帧差图
            diff = np.abs(frame2_rgb.astype(int) - frame1_rgb.astype(int)).astype(np.uint8)
            img_diff = Image.fromarray(diff)
            img_diff.save(str(out_dir / f"diff_{pairs_saved:04d}.png"))

            pairs_saved += 1
            total_pairs += 1

        cap.release()
        print(f"  ✅ 提取了 {pairs_saved} 对帧")

    print("\n" + "=" * 60)
    print(f"✅ 共提取 {total_pairs} 对帧到 {output_dir}")
    print("\n📊 使用这些帧对训练帧差 LoRA:")
    print("   python train_video_lora.py --mode frame_diff --video_dir data/framediff")
    return total_pairs


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='增强版预处理脚本',
        formatter_class=argparse.RawTextHelp
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['images', 'temporal', 'framediff'],
                        help="""处理模式:
  images     - 静态图片 (普通 LoRA)
  temporal   - 时序序列 (Temporal LoRA)
  framediff  - 帧差对 (Frame Diff LoRA)""")
    parser.add_argument('--input', type=str, required=True,
                        help='输入视频目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--fps', type=float, default=8,
                        help='抽帧目标帧率')
    parser.add_argument('--resolution', type=str, default='512x512',
                        help='分辨率 (WxH)')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='序列长度（帧数），仅 temporal 模式')
    parser.add_argument('--max_frames', type=int, default=30,
                        help='每个视频最大提取帧数')

    args = parser.parse_args()

    resolution = tuple(map(int, args.resolution.split('x')))

    print(f"🎬 增强预处理")
    print(f"   模式: {args.mode}")
    print(f"   输入: {args.input}")
    print(f"   输出: {args.output}")
    print("=" * 60)

    if args.mode == 'images':
        count = extract_images(
            video_dir=args.input,
            output_dir=args.output,
            fps_target=args.fps,
            resolution=resolution,
            max_frames=args.max_frames,
        )
    elif args.mode == 'temporal':
        count = extract_temporal(
            video_dir=args.input,
            output_dir=args.output,
            seq_length=args.seq_length,
            fps_target=args.fps,
            resolution=resolution,
        )
    elif args.mode == 'framediff':
        count = extract_frame_diff(
            video_dir=args.input,
            output_dir=args.output,
            resolution=tuple(int(x) for x in ('256,256').split(',')),
        )


if __name__ == '__main__':
    main()
