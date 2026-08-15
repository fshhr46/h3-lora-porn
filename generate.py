#!/usr/bin/env python3
"""
使用训练好的 LoRA 权重生成测试视频
支持导出为 H3 平台可使用的格式

使用方法:
    python generate.py --lora_path output/models/final_lora --prompt "beautiful woman in silk dress" --output output/videos
"""

import argparse
import os
import sys
import json
from pathlib import Path

try:
    import torch
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"缺少依赖: {e}")
    sys.exit(1)


# ============================================================
# LoRA 加载与推理
# ============================================================

class LoRAImageGenerator:
    """LoRA 图像生成器（用于快速测试）"""

    def __init__(self, base_model, lora_path, device='cuda'):
        self.device = torch.device(device)
        print(f"🖥️  设备: {self.device}")

        # 加载基础管道
        print(f"📦 加载基础模型: {base_model}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        )
        self.pipe.to(self.device)

        # 加载 LoRA
        print(f"🎯 加载 LoRA: {lora_path}")
        self.lora_path = Path(lora_path)

        # 加载配置
        config_path = self.lora_path / "lora_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            print(f"   配置: {json.dumps(self.config, indent=2)}")
        else:
            self.config = {}

        # 加载 LoRA 权重
        self.lora_state_dict = torch.load(
            self.lora_path / "pytorch_lora_weights.bin",
            map_location='cpu'
        )

        # 注入 LoRA 到 UNet
        self._inject_lora()

        print("✅ 加载完成")

    def _inject_lora(self):
        """将 LoRA 权重注入到 UNet"""
        # 方法 1: 使用 diffusers 的 load_lora_weights
        try:
            self.pipe.load_lora_weights(
                str(self.lora_path.parent),
                weight_name="pytorch_lora_weights.bin",
                adapter_name="lora_adapter"
            )
            print("   使用 diffusers load_lora_weights 加载成功")
        except Exception:
            # 方法 2: 手动注入
            from diffusers.models.attention_processor import LoRAAttnProcessor
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

            # 遍历 UNet 的 attention 层
            attn_modules = []
            for module in self.pipe.unet.modules():
                module_type = type(module).__name__
                if any(keyword in module_type for keyword in ['CrossAttn', 'Attn']):
                    attn_modules.append(module)

            if not attn_modules:
                print("   ⚠️  未找到注意力模块，尝试其他方法")
                return

            # 为每个注意力模块添加 LoRA
            lora_rank = self.config.get('lora_rank', 32)
            lora_alpha = self.config.get('lora_alpha', 16)

            for module in attn_modules:
                if hasattr(module, 'lora_up') or hasattr(module, 'to_q_lora'):
                    continue  # 已经注入了

                lora_proc = LoRAAttnProcessor(
                    hidden_size=module.to_q.out_features,
                    cross_attention_dim=module.to_k.in_features,
                    rank=lora_rank,
                )
                module.set_processor(lora_proc)

            # 加载权重
            self.pipe.unet.load_state_dict(
                {k.replace('.processor.', '.'): v
                 for k, v in self.lora_state_dict.items()},
                strict=False
            )
            print("   使用手动注入加载成功")

    def generate(self, prompt, negative_prompt="", lora_weight=0.7,
                 width=512, height=512, num_steps=30, seed=None):
        """生成单张图像"""

        # 设置 LoRA 权重
        scale = lora_weight

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        print(f"\n🎨 生成中...")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   LoRA 权重: {scale}")
        print(f"   步数: {num_steps}")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            width=width,
            height=height,
            generator=generator,
            cross_attention_scale=scale,
        ).images[0]

        return image

    def generate_batch(self, prompts, output_dir, lora_weight=0.7, **kwargs):
        """批量生成"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(prompts)}] {prompt}")

            image = self.generate(prompt, lora_weight=lora_weight, **kwargs)

            # 保存
            filename = f"lora_test_{i+1:04d}.png"
            filepath = output_path / filename
            image.save(str(filepath))
            print(f"   💾 已保存: {filepath}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LoRA 测试生成')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='LoRA 模型路径')
    parser.add_argument('--base_model', type=str,
                        default='stabilityai/stable-diffusion-xl-base-1.0',
                        help='基础模型')
    parser.add_argument('--prompts', type=str,
                        default='prompts/positive_prompts.txt',
                        help='提示词文件路径')
    parser.add_argument('--output', type=str, default='output/videos',
                        help='输出目录')
    parser.add_argument('--lora_weight', type=float, default=0.7,
                        help='LoRA 权重（0.0-1.0）')
    parser.add_argument('--num_steps', type=int, default=30,
                        help='采样步数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备')
    args = parser.parse_args()

    # 加载提示词
    prompt_path = Path(args.prompts)
    if prompt_path.exists() and prompt_path.is_file():
        with open(prompt_path) as f:
            prompts = [line.strip() for line in f
                       if line.strip() and not line.startswith('#')]
        print(f"📝 从 {prompt_path} 加载了 {len(prompts)} 个提示词")
    else:
        prompts = ["beautiful woman, cinematic lighting, silk dress"]
        print(f"⚠️  未找到提示词文件，使用默认提示词")

    # 初始化生成器
    generator = LoRAImageGenerator(args.base_model, args.lora_path, args.device)

    # 批量生成
    generator.generate_batch(
        prompts=prompts,
        output_dir=args.output,
        lora_weight=args.lora_weight,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("✅ 生成完成！")
    print(f"   输出目录: {args.output}")
    print(f"   LoRA 权重建议:")
    print(f"   - 效果不够明显 → 提高到 0.8-1.0")
    print(f"   - 过拟合/失真 → 降低到 0.3-0.5")
    print("=" * 60)


if __name__ == '__main__':
    main()
