#!/bin/bash
# ============================================
# LoRA 训练一键脚本 - MiniMax H3 视频模型
# ============================================

set -e

echo "============================================"
echo "  LoRA 训练脚本 - H3 Video Model"
echo "============================================"

# ---------- 配置 ----------
BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
# 或 SD1.5: runwayml/stable-diffusion-v1-5
# 或 SDXL:   stabilityai/stable-diffusion-xl-base-1.0

LORA_RANK=32
LORA_ALPHA=16
EPOCHS=15
BATCH_SIZE=4
LEARNING_RATE=1e-4
RESOLUTION="512,512"
OUTPUT_DIR="output/models"
DATA_DIR="data/filtered"
GRADIENT_ACCUMULATION=4
MAX_STEPS=0
MIXED_PRECISION="fp16"
SEED=42

# ---------- 检查依赖 ----------
echo "[1/6] 检查依赖..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip 未安装"
    exit 1
fi

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 可用"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  未检测到 NVIDIA GPU，将使用 CPU 训练（较慢）"
fi

# ---------- 安装依赖 ----------
echo "[2/6] 安装/更新依赖..."
pip install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

# ---------- 数据检查 ----------
echo "[3/6] 检查训练数据..."

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "⚠️  数据目录 $DATA_DIR 为空或不存在"
    echo "   请先将训练素材放入 data/filtered/ 目录"
    echo "   参考 README.md 中的素材收集指南"
    read -p "是否继续（使用空数据）？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

DATA_COUNT=$(find "$DATA_DIR" -type f | wc -l)
echo "   找到 $DATA_COUNT 个训练文件"

if [ "$DATA_COUNT" -lt 10 ]; then
    echo "⚠️  素材数量较少（少于10个），建议至少准备 30-100 个素材"
    read -p "是否继续？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ---------- 创建输出目录 ----------
echo "[4/6] 创建输出目录..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "output/videos"

# ---------- 开始训练 ----------
echo "[5/6] 开始训练..."
echo "   基础模型: $BASE_MODEL"
echo "   LoRA 秩: $LORA_RANK"
echo "   训练轮数: $EPOCHS"
echo "   批次大小: $BATCH_SIZE"
echo "   学习率: $LEARNING_RATE"
echo "   分辨率: ${RESOLUTION//,/x}"
echo "   混合精度: $MIXED_PRECISION"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"

# ---------- 训练命令（根据实际框架调整）----------
# 使用 diffusers/PEFT 训练 LoRA:
python train_lora.py \
    --base_model "$BASE_MODEL" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --resolution $RESOLUTION \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mixed_precision $MIXED_PRECISION \
    --seed $SEED \
    --max_steps $MAX_STEPS \
    2>&1 | tee "$OUTPUT_DIR/training_log.txt"

# ---------- 完成 ----------
echo "[6/6] 训练完成！"
echo "============================================"
echo "✅ LoRA 模型已保存到: $OUTPUT_DIR/"
echo ""
echo "下一步："
echo "1. 前往 output/videos/ 查看生成测试结果"
echo "2. 使用 prompts/video_prompts.txt 中的提示词"
echo "3. 在 H3 平台上加载 LoRA 进行视频生成"
echo "4. LoRA 权重从 0.5 开始尝试，逐步调整"
echo "============================================"
