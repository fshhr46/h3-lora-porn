#!/usr/bin/env python3
"""
LoRA 训练主脚本 - 用于训练视频模型 LoRA 权重
支持 SD/SDXL 扩散模型的 LoRA 训练
"""

import argparse
import os
import sys
import random
import json
import glob
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请先运行: pip install -r requirements.txt")
    sys.exit(1)


# ============================================================
# 数据集类
# ============================================================

class VideoFrameDataset(Dataset):
    """视频帧数据集，用于 LoRA 训练"""

    def __init__(self, data_dir, resolution=(512, 512), tokenizer=None):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.images = []
        self.captions = []

        # 支持的图片格式
        supported = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

        # 扫描数据目录
        for ext in supported:
            for fpath in self.data_dir.rglob(f'*{ext}'):
                self.images.append(fpath)

        # 尝试加载 caption 文件（如果有）
        caption_dir = self.data_dir.parent / 'captions'
        self.captions = []
        for img_path in self.images:
            stem = img_path.stem
            cap_path = caption_dir / f"{stem}.txt"
            if cap_path.exists():
                self.captions.append(cap_path.read_text().strip())
            else:
                # 如果没有 caption，使用默认描述
                self.captions.append("beautiful woman, cinematic lighting")

        if len(self.images) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何图片")

        print(f"✅ 加载了 {len(self.images)} 个训练样本")

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        """加载并预处理图像"""
        try:
            img = Image.open(path).convert('RGB')
            # 中心裁剪到目标分辨率
            w, h = img.size
            target_w, target_h = self.resolution
            if w / h < target_w / target_h:
                new_w = w
                new_h = int(w * target_h / target_w)
            else:
                new_h = h
                new_w = int(h * target_w / target_h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 中心裁剪
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            img = img.crop((left, top, left + target_w, top + target_h))

            return img
        except Exception as e:
            print(f"⚠️  无法加载图像 {path}: {e}")
            # 返回黑色占位图
            return Image.new('RGB', self.resolution, (0, 0, 0))

    def __getitem__(self, idx):
        image = self._load_image(self.images[idx])
        caption = self.captions[idx]

        # 数据增强：随机水平翻转
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 归一化为 tensor
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
        image_tensor = image_tensor / 127.5 - 1.0  # [-1, 1]
        image_tensor = image_tensor.permute(2, 0, 1)  # CHW

        # Tokenize caption
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt'
            ).input_ids.squeeze()
        else:
            input_ids = torch.zeros(self.tokenizer.model_max_length if self.tokenizer else 77, dtype=torch.long)

        return {
            'pixel_values': image_tensor,
            'input_ids': input_ids,
            'caption': caption,
        }


# ============================================================
# LoRA 训练器
# ============================================================

class LoRATrainer:
    """LoRA 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🖥️  设备: {self.device}")

        # 加载 tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")

        # 加载文本编码器（冻结）
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.base_model, subfolder="text_encoder"
        ).to(self.device).eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 加载 UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            args.base_model, subfolder="unet"
        ).to(self.device)

        # 配置 LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # CrossAttention 投影
                "query", "key", "value", "out",       # 自注意力投影
                "conv1", "conv2",                      # Conv 层
                "conv_shortcut",                       # 捷径连接
            ],
            lora_dropout=0.1,
        )

        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()

        # 加载数据集
        self.dataset = VideoFrameDataset(
            args.data_dir,
            resolution=tuple(map(int, args.resolution.split(','))),
            tokenizer=self.tokenizer
        )

        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # 优化器
        trainable_params = [
            p for p in self.unet.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=1e-2,
        )

        # 调度器
        total_steps = len(self.dataloader) * args.epochs
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # 保存路径
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        """执行训练"""
        print("\n🚀 开始训练...")
        print(f"   总步数: {len(self.dataloader) * self.args.epochs}")
        print(f"   每个 epoch: {len(self.dataloader)} 步")
        print("=" * 60)

        best_loss = float('inf')

        for epoch in range(self.args.epochs):
            self.unet.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.dataloader):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)

                # 前向传播
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                        text_embeddings = self.text_encoder(input_ids)[0]

                with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                    # 随机噪声
                    noise = torch.randn_like(pixel_values)
                    # 随机 timestep
                    timesteps = torch.randint(
                        0, 1000, (pixel_values.shape[0],),
                        device=self.device
                    ).long()

                    # UNet 预测噪声
                    noise_pred = self.unet(
                        pixel_values, timesteps, encoder_hidden_states=text_embeddings
                    ).sample

                    # MSE Loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(f"  Epoch {epoch+1}/{self.args.epochs} | "
                          f"Batch {batch_idx}/{len(self.dataloader)} | "
                          f"Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"\n✅ Epoch {epoch+1}/{self.args.epochs} | "
                  f"Average Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(f"best_lora")

            # 定期保存检查点
            if (epoch + 1) % self.args.save_every == 0:
                self.save(f"lora_epoch_{epoch+1}")

            print("-" * 60)

        # 最终保存
        self.save("final_lora")
        print("\n🎉 训练完成！模型已保存到:", self.output_dir)

    def save(self, name="lora"):
        """保存 LoRA 权重"""
        save_path = self.output_dir / f"{name}"
        self.unet.save_pretrained(str(save_path))

        # 保存训练配置
        config = {
            'base_model': self.args.base_model,
            'lora_rank': self.args.lora_rank,
            'lora_alpha': self.args.lora_alpha,
            'resolution': self.args.resolution,
            'epochs': self.args.epochs,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'seed': self.args.seed,
        }
        with open(self.output_dir / f"{name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 已保存: {save_path}")


# ============================================================
# 主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='LoRA 训练脚本')

    # 模型
    parser.add_argument('--base_model', type=str, default='stabilityai/stable-diffusion-xl-base-1.0',
                        help='基础模型路径')
    parser.add_argument('--data_dir', type=str, default='data/filtered',
                        help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='output/models',
                        help='输出目录')

    # LoRA 参数
    parser.add_argument('--lora_rank', type=int, default=32,
                        help='LoRA 秩 (8/16/32/64)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--resolution', type=str, default='512,512',
                        help='分辨率 (W,H)')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    trainer = LoRATrainer(args)
    trainer.train()
