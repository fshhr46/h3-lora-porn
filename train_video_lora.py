#!/usr/bin/env python3
"""
视频帧序列训练模块 - 用于训练时序/动态 LoRA
支持训练模型的"运动模式"而非仅静态外观

原理:
- 从视频中提取有序帧序列
- 训练时让模型学习帧之间的时序关系
- 生成时能输出更自然的动态效果

支持格式:
- AnimateDiff 兼容 (Motion LoRA)
- 帧差训练 (Frame Diff LoRA)
- 光流辅助训练 (Optical Flow LoRA)
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV 未安装，部分功能不可用")


# ============================================================
# 视频帧序列数据集
# ============================================================

class VideoSequenceDataset(Dataset):
    """视频帧序列数据集
    
    从视频中提取连续的帧序列，用于训练时序 LoRA
    """

    def __init__(
        self,
        video_dir: str,
        seq_length: int = 8,        # 每个序列的帧数
        step: int = 1,              # 帧步长
        resolution: Tuple[int, int] = (512, 512),
        fps_target: float = 8.0,
        use_optical_flow: bool = False,
    ):
        self.video_dir = Path(video_dir)
        self.seq_length = seq_length
        self.step = step
        self.resolution = resolution
        self.fps_target = fps_target
        self.use_optical_flow = use_optical_flow

        # 找到所有视频文件
        self.videos = []
        for ext in ('.mp4', '.avi', '.mov', '.webm'):
            self.videos.extend(self.video_dir.rglob(f'*{ext}'))

        if len(self.videos) == 0:
            raise FileNotFoundError(f"在 {video_dir} 中未找到任何视频")

        print(f"✅ 找到 {len(self.videos)} 个视频文件")

    def __len__(self):
        return len(self.videos)

    def _read_video_frames(self, video_path: Path, num_frames: int) -> List[np.ndarray]:
        """读取视频并提取等间隔帧"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            cap.release()
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return []

        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB, resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, self.resolution)
            frames.append(frame_rgb)

        cap.release()
        return frames

    def _compute_optical_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """计算帧间光流"""
        flows = []
        prev_frame = None

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0
                )
                # 归一化光流向量
                flow_norm = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                flows.append(flow_norm)
            prev_frame = frame

        return flows

    def __getitem__(self, idx):
        video_path = self.videos[idx]

        # 从单个视频中提取多个序列
        frames = self._read_video_frames(video_path, self.seq_length * 3)  # 多提取一些

        if len(frames) < self.seq_length:
            # 不够帧，返回占位
            placeholder = np.zeros((*self.resolution, 3), dtype=np.float32)
            return {
                'frames': torch.zeros(self.seq_length, 3, *self.resolution),
                'optical_flow': torch.zeros(self.seq_length, *self.resolution) if self.use_optical_flow else None,
                'video_name': video_path.name,
            }

        # 随机选择一个起始位置
        start_idx = np.random.randint(0, max(1, len(frames) - self.seq_length))
        seq_frames = frames[start_idx:start_idx + self.seq_length]

        # 归一化到 [-1, 1]
        frames_tensor = torch.tensor(np.array(seq_frames), dtype=torch.float32)
        frames_tensor = frames_tensor / 127.5 - 1.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # TCHW

        # 计算光流（可选）
        optical_flow = None
        if self.use_optical_flow:
            flows = self._compute_optical_flow(seq_frames)
            if len(flows) > 0:
                optical_flow = torch.tensor(np.array(flows), dtype=torch.float32)

        return {
            'frames': frames_tensor,
            'optical_flow': optical_flow,
            'video_name': video_path.name,
        }


# ============================================================
# 时序 LoRA 训练器
# ============================================================

class TemporalLoRATrainer:
    """时序 LoRA 训练器
    
    训练视频帧序列之间的转换关系
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🖥️  设备: {self.device}")

        # 加载数据集
        self.dataset = VideoSequenceDataset(
            args.video_dir,
            seq_length=args.seq_length,
            step=args.step,
            resolution=tuple(map(int, args.resolution.split(','))),
            fps_target=args.fps,
            use_optical_flow=args.use_optical_flow,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # 初始化时序编码器
        self._init_models()

        # 优化器
        trainable_params = [
            p for p in self.models.values()
            for p in p.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=1e-2,
        )

        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_models(self):
        """初始化时序模型"""
        from diffusers import MotionAdapter, AnimatedDiffPipeline

        print("📦 加载 Motion Adapter (AnimateDiff)...")

        # Motion Adapter: 处理视频时序
        self.models['motion_adapter'] = MotionAdapter().to(self.device)

        # 时序 LoRA 层
        self.models['temporal_lora'] = nn.ModuleDict({
            'conv1d': nn.Conv1d(
                in_channels=320,
                out_channels=320,
                kernel_size=3,
                padding=1,
            ),
            'temporal_attn': nn.MultiheadAttention(
                embed_dim=320,
                num_heads=8,
                batch_first=True,
            ),
        }).to(self.device)

        # 冻结 Motion Adapter，只训练 LoRA
        for param in self.models['motion_adapter'].parameters():
            param.requires_grad = False

        print("   可训练参数:")
        for name, model in self.models.items():
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"   - {name}: {trainable}/{total} trainable")

    def train(self):
        """执行训练"""
        print("\n🚀 开始时序 LoRA 训练...")

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in self.dataloader:
                frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
                batch_size = frames.size(0)

                self.models['temporal_lora'].train()

                # 前向传播
                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # 1. 帧编码
                    encoded_frames = []
                    for t in range(frames.size(1)):
                        frame = frames[:, t]  # [B, C, H, W]
                        # 简化的编码器（实际使用 UNet）
                        encoded = F.interpolate(
                            frame,
                            size=(64, 64),
                            mode='bilinear',
                            align_corners=False,
                        )
                        encoded = F.avg_pool2d(encoded, 2)  # [B, C, 32, 32]
                        encoded_frames.append(encoded)

                    encoded_seq = torch.stack(encoded_frames, dim=1)  # [B, T, C, H, W]

                    # 2. 时序 LoRA 处理
                    b, t, c, h, w = encoded_seq.shape
                    # 展平成序列
                    encoded_seq = encoded_seq.permute(0, 2, 1, 3, 4).reshape(b * c, t, h * w)

                    # Conv1d + Temporal Attention
                    x = self.models['temporal_lora']['conv1d'](encoded_seq)
                    x = F.relu(x)

                    x = x.permute(0, 2, 1)  # [bc, hw, t]
                    attn_out, _ = self.models['temporal_lora']['temporal_attn'](x, x, x)
                    attn_out = attn_out.permute(0, 2, 1)

                    # 3. 重建
                    reconstructed = x + attn_out

                    # 4. 损失：帧间一致性 + 时序平滑
                    frame_diff_loss = 0.0
                    for t_idx in range(1, t):
                        diff = encoded_seq[:, :, t_idx] - encoded_seq[:, :, t_idx-1]
                        frame_diff_loss += torch.mean(diff ** 2)

                    temporal_smooth_loss = torch.mean(reconstructed[:, 1:] - reconstructed[:, :-1]) ** 2

                    loss = frame_diff_loss + 0.1 * temporal_smooth_loss

                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.args.epochs} | "
                          f"Batch {num_batches} | "
                          f"Loss: {loss.item():.4f}")

            avg_loss = total_loss / max(num_batches, 1)
            print(f"\n✅ Epoch {epoch+1}/{self.args.epochs} | "
                  f"Average Loss: {avg_loss:.4f}")

            # 定期保存
            if (epoch + 1) % self.args.save_every == 0 or epoch == self.args.epochs - 1:
                self.save(f"temporal_lora_epoch_{epoch+1}")

            print("-" * 60)

        print("\n🎉 时序 LoRA 训练完成！")

    def save(self, name="temporal_lora"):
        """保存时序 LoRA 权重"""
        save_path = self.output_dir / name

        for module_name, module in self.models.items():
            path = save_path / f"{module_name}.pth"
            torch.save(module.state_dict(), str(path))

        # 保存配置
        config = {
            'seq_length': self.args.seq_length,
            'step': self.args.step,
            'resolution': self.args.resolution,
            'fps': self.args.fps,
            'use_optical_flow': self.args.use_optical_flow,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 已保存: {save_path}")


# ============================================================
# 帧差 LoRA（更简单直接的方法）
# ============================================================

class FrameDiffLoRATrainer:
    """帧差 LoRA 训练器
    
    直接学习相邻帧之间的差异，用于视频增强
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 帧差 LoRA 层（比时序 LoRA 更轻量）
        self.diff_lora = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),  # 输出帧差 [-1, 1]
        ).to(self.device)

        trainable_params = list(self.diff_lora.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """训练帧差 LoRA"""
        print("🚀 开始帧差 LoRA 训练...")

        # 读取视频序列作为训练数据
        video_files = list(Path(self.args.video_dir).rglob('*.mp4'))
        if not video_files:
            print("❌ 未找到视频文件")
            return

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            num_batches = 0

            # 随机选择一个视频
            video_path = video_files[np.random.randint(len(video_files))]
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for batch_idx in range(self.args.batch_size):
                # 随机选择起始帧
                start = np.random.randint(0, max(1, total_frames - 2))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                ret1, frame1 = cap.read()
                ret2, frame2 = cap.read()

                if not ret1 or not ret2:
                    continue

                # BGR -> RGB, resize
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame1 = cv2.resize(frame1, (self.args.frame_w, self.args.frame_h))
                frame2 = cv2.resize(frame2, (self.args.frame_w, self.args.frame_h))

                # 归一化
                frame1_tensor = torch.tensor(frame1 / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)
                frame2_tensor = torch.tensor(frame2 / 127.5 - 1.0, dtype=torch.float32).permute(2, 0, 1)

                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # 预测帧差
                    predicted_diff = self.diff_lora(frame1_tensor.unsqueeze(0))

                    # 重建 frame2
                    reconstructed = frame2_tensor.unsqueeze(0)
                    actual_diff = reconstructed - frame1_tensor.unsqueeze(0)

                    # MSE Loss
                    loss = nn.functional.mse_loss(predicted_diff, actual_diff)

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.optimizer.zero_grad()
                total_loss += loss.item()
                num_batches += 1

            cap.release()

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Epoch {epoch+1}/{self.args.epochs} | Loss: {avg_loss:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == self.args.epochs - 1:
                self.save(f"frame_diff_lora_epoch_{epoch+1}")

        print("🎉 帧差 LoRA 训练完成！")

    def save(self, name="frame_diff_lora"):
        save_path = self.output_dir / name
        torch.save(self.diff_lora.state_dict(), str(save_path / "pytorch_model.bin"))

        config = {
            'frame_w': self.args.frame_w,
            'frame_h': self.args.frame_h,
            'epochs': self.args.epochs,
            'learning_rate': self.args.learning_rate,
        }
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='高级视频序列 LoRA 训练',
        formatter_class=argparse.RawTextHelp
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['temporal', 'frame_diff'],
                        help="""训练模式:
  temporal     - 时序 LoRA（学习帧间运动模式，适合 AnimateDiff）
  frame_diff   - 帧差 LoRA（学习相邻帧差异，轻量快速）""")
    parser.add_argument('--video_dir', type=str, required=True,
                        help='视频素材目录')
    parser.add_argument('--output_dir', type=str, default='output/models',
                        help='输出目录')
    parser.add_argument('--resolution', type=str, default='512,512',
                        help='分辨率 (W,H)')

    # Temporal LoRA 参数
    parser.add_argument('--seq_length', type=int, default=8,
                        help='序列长度（帧数），仅 temporal 模式')
    parser.add_argument('--step', type=int, default=1,
                        help='帧采样步长')
    parser.add_argument('--fps', type=float, default=8.0,
                        help='目标帧率')
    parser.add_argument('--use_optical_flow', action='store_true',
                        help='是否使用光流辅助训练')

    # 通用参数
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每 N 轮保存检查点')
    parser.add_argument('--frame_w', type=int, default=256,
                        help='帧差模式宽度')
    parser.add_argument('--frame_h', type=int, default=256,
                        help='帧差模式高度')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == 'temporal':
        trainer = TemporalLoRATrainer(args)
        trainer.train()
    elif args.mode == 'frame_diff':
        trainer = FrameDiffLoRATrainer(args)
        trainer.train()


if __name__ == '__main__':
    main()
