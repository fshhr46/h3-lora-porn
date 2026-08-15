# MiniMax H3 视频模型 LoRA 训练全流程

> 目标：训练一个 LoRA 权重，让 H3（海螺视频模型）生成高质量的色情/半裸女性视频。

## 📁 项目结构

```
├── data/
│   ├── raw/              # 原始下载的视频素材
│   ├── cropped/          # 静态图片（普通 LoRA 训练）
│   ├── temporal/         # 时序序列（Temporal LoRA 训练）
│   ├── framediff/        # 帧差对（Frame Diff LoRA 训练）
│   └── filtered/         # 人工筛选后的高质量素材
├── output/
│   ├── models/           # 训练好的 LoRA 模型
│   └── videos/           # 生成的测试视频
├── train.sh              # 一键训练脚本（含高级模式）
├── collect.py            # 素材收集脚本（支持成人网站）
├── preprocess.py         # 基础预处理（抽帧、裁剪）
├── preprocess_enhanced.py# 增强预处理（支持时序/帧差提取）
├── train_lora.py         # 普通 LoRA 训练（静态图片）
├── train_video_lora.py   # 高级视频 LoRA 训练（时序/帧差）
├── generate.py           # LoRA 测试生成
├── train_config.yaml     # 训练配置
├── prompts/              # 提示词模板
│   ├── positive_prompts.txt
│   ├── negative_prompts.txt
│   └── video_prompts.txt
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### Step 1: 环境准备

```bash
pip install -r requirements.txt
```

### Step 2: 收集训练素材

```bash
python collect.py --source youtube --keywords "beautiful woman,bikini,lingerie,silky dress" --count 50
python collect.py --source twitter --keywords "anime girl,swimsuit,wet dress" --count 50
python collect.py --source reddit --subreddit "r/facelesspods r/undressapp" --count 50
```

### Step 3: 预处理

**选择训练模式：**

```bash
# 模式 A: 普通 LoRA（静态图片）— 入门
python preprocess_enhanced.py --mode images --input data/raw --output data/cropped --fps 8

# 模式 B: Temporal LoRA（时序序列）— 进阶
python preprocess_enhanced.py --mode temporal --input data/raw --output data/temporal --fps 8 --seq_length 8

# 模式 C: Frame Diff LoRA（帧差对）— 进阶
python preprocess_enhanced.py --mode framediff --input data/raw --output data/framediff
```

### Step 4: 人工筛选

将 `data/cropped/` 中不满意的视频移入 `data/filtered/`（反向操作也可）。

### Step 5: 训练 LoRA

**选择训练模式：**

```bash
# 模式 A: 普通 LoRA（静态图片）
bash train.sh

# 模式 B: Temporal LoRA（时序序列）
python train_video_lora.py --mode temporal --video_dir data/temporal --output_dir output/models/temporal --epochs 20

# 模式 C: Frame Diff LoRA（帧差对）
python train_video_lora.py --mode frame_diff --video_dir data/framediff --output_dir output/models/framediff --epochs 20
```

### Step 6: 测试生成

使用训练好的 LoRA 在 H3 平台上生成视频，提示词见 `prompts/`。

---

## 📊 素材收集指南

### 🌟 成人/色情素材来源（推荐优先）

#### 1. RedGifs ⭐⭐⭐⭐⭐（最佳）
- **网址**: https://www.redgifs.com
- **特点**: 短视频/GIF，完美适合训练，无水印
- **搜索词**: `bikini`, `lingerie`, `wet`, `swimsuit`, `thong`, `sheer`, `gravure`
- **命令**:
  ```bash
  python collect.py --source redgifs --keywords "bikini,lingerie,wet,swimsuit,thong,gravure" --count 30
  ```

#### 2. Xvideos ⭐⭐⭐⭐
- **网址**: https://www.xvideos.com
- **特点**: 量大，慢动作视频多
- **搜索词**: `slow motion`, `bikini`, `wet`, `swimsuit`, `lingerie`
- **命令**:
  ```bash
  python collect.py --source xvideos --keywords "bikini slow motion,wet swimsuit,lingerie" --count 30
  ```

#### 3. SpankWire ⭐⭐⭐⭐
- **网址**: https://www.spankwire.com
- **特点**: 短片段丰富，质量高
- **命令**:
  ```bash
  python collect.py --source spankwire --keywords "bikini,wet,silky dress,lingerie" --count 30
  ```

#### 4. Reddit 成人板块 ⭐⭐⭐⭐
| Subreddit | 内容 | 命令 |
|-----------|------|------|
| r/facelesspods | 无脸女性视频 | `--source reddit --subreddit facelesspods` |
| r/wallsofsluts | 各种场景 | `--source reddit --subreddit wallsofsluts` |
| r/underboob | 胸部特写 | `--source reddit --subreddit underboob` |
| r/thighhighs | 过膝袜 | `--source reddit --subreddit thighhighs` |
| r/wetandteasing | 湿身 | `--source reddit --subreddit wetandteasing` |
| r/animatedporn | 动漫色情 | `--source reddit --subreddit animatedporn` |
| r/lewdanimemes | 动漫表情包 | `--source reddit --subreddit lewdanimemes` |

### 其他来源

#### YouTube
- 搜索关键词：`"beautiful woman slow motion"`, `"lingerie photoshoot"`, `"wet dress slow motion"`, `"anime girl dancing"`, `"silky dress cinematic"`

#### Twitter/X
- 搜索标签：`#slowmotion` `#bikini` `#lingerie` `#gravure` `#swimsuit` `#wet` `#thong` `#sheer`

#### 免费素材网站
- Pexels (pexels.com) — 免费商用
- Pixabay (pixabay.com) — 免费量大
- Mixkit (mixkit.co) — 电影感素材

#### Gravure 网站
- gravure365.jp — 日本写真偶像
- Manapla.com — 偶像写真

4. **Pexels/Pixabay** — 免费素材网站

### 素材标准

- ✅ 分辨率 ≥ 720P
- ✅ 帧率 ≥ 24fps（越高越好）
- ✅ 内容稳定（不要频繁切换场景）
- ✅ 光线好，主体清晰
- ✅ 时长 3-10 秒最佳
- ❌ 避免快速镜头切换
- ❌ 避免模糊/低画质
- ❌ 避免多人/杂乱背景

---

## 🎯 训练配置说明

参见 `train_config.yaml`，关键参数：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| base_model | stabilityai/stable-diffusion-xl-base-1.0 | 基础模型 |
| lora_rank | 32 | LoRA 秩，越大越强但容易过拟合 |
| learning_rate | 1e-4 | 学习率 |
| epochs | 10-20 | 训练轮数 |
| resolution | 512x512 | 训练分辨率 |
| batch_size | 4 | 批次大小 |
| seed | 42 | 随机种子 |

---

## 💬 提示词模板

### 正向提示词
```
beautiful woman, cinematic lighting, slow motion,
soft focus, [风格描述], [场景描述]
```

### 负向提示词
```
low quality, blurry, distorted, bad anatomy,
extra limbs, watermark, text
```

### 视频生成提示词
见 `prompts/video_prompts.txt`

---

## ⚙️ 硬件要求

- **GPU**: NVIDIA RTX 3060 以上（12GB VRAM 推荐）
- **RAM**: 16GB 以上
- **磁盘**: 至少 20GB 可用空间
- **训练时间**: 4-12 小时（取决于素材数量和 GPU）

---

## 📝 注意事项

1. **素材质量 > 数量** — 100 张高质量素材胜过 500 张一般素材
2. **避免过拟合** — 如果训练后所有输出都一样，减少 epochs 或降低 rank
3. **LoRA 权重** — 在 H3 上测试时，从 0.5 开始逐步调高权重
4. **风格一致性** — 所有素材应尽量保持一致的风格（真人/动漫/插画）
5. **版权注意** — 训练素材用于生成视频时注意版权问题

---

## 🚀 高级：视频序列 LoRA 训练

### 为什么需要？

普通 LoRA 只用静态图片训练，学会的是**外观风格**。
视频序列 LoRA 学习的是**运动模式**和**帧间关系**，能生成更自然的动态效果。

### 三种训练模式对比

| 模式 | 训练数据 | 学习什么 | 难度 | 效果 |
|------|----------|----------|------|------|
| **普通 LoRA** | 静态图片 | 外观/风格/服装 | ⭐ | 静态风格强，动态一般 |
| **Temporal LoRA** | 帧序列 (8帧) | 运动模式/时序关系 | ⭐⭐⭐ | 动态更自然 |
| **Frame Diff LoRA** | 相邻帧对 | 帧间变化 | ⭐⭐ | 轻量，适合微调 |

### Temporal LoRA 原理

```
视频: [帧1][帧2][帧3][帧4][帧5][帧6][帧7][帧8]
            ↓ 抽帧 + 排序
序列: {frame_0, frame_1, ..., frame_7}
            ↓ 训练
时序 LoRA 权重 → 学习 frame_i → frame_{i+1} 的转换
            ↓ 推理时
给定初始帧 + 提示词 → 模型预测后续帧 → 生成视频
```

### Frame Diff LoRA 原理

```
视频帧对: (frame_A, frame_B)
            ↓
帧差 = frame_B - frame_A
            ↓ 训练
帧差 LoRA → 学习如何从 A 变换到 B
            ↓ 推理时
输入帧 A + 帧差 LoRA → 预测帧 B → 拼接成视频
```

### 使用示例

```bash
# 1. 提取时序序列
python preprocess_enhanced.py --mode temporal \
    --input data/raw \
    --output data/temporal \
    --fps 8 \
    --seq_length 8

# 2. 训练时序 LoRA
python train_video_lora.py --mode temporal \
    --video_dir data/temporal \
    --output_dir output/models/temporal \
    --epochs 20 \
    --seq_length 8 \
    --batch_size 2 \
    --learning_rate 1e-4

# 3. 或使用帧差模式
python preprocess_enhanced.py --mode framediff \
    --input data/raw \
    --output data/framediff

python train_video_lora.py --mode frame_diff \
    --video_dir data/framediff \
    --output_dir output/models/framediff \
    --epochs 20
```

### 混合使用（推荐）

```
1. 先用普通 LoRA 训练外观风格
2. 再用 Temporal LoRA 训练运动模式
3. 在 H3 上同时加载两种 LoRA
4. 效果最佳
```
