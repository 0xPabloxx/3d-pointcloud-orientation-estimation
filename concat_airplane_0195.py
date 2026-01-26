"""
将所有 airplane 0195 图片从左到右拼接起来，并添加标题
统一极坐标图尺寸
"""
from PIL import Image, ImageDraw, ImageFont
import os

# 定义所有图片的路径（按顺序）和对应的标题
image_paths = [
    # D_16a
    ("Discrete", "paper_figures/d_series_vis/D_16a/airplane_airplane_0195_D_16a.png"),
    # MF full
    ("MvM", "paper_figures/mf_full/airplane_airplane_0195_mf.png"),
    # P2V2 experts
    ("MoE", "paper_figures/p2v2_experts_all/airplane_airplane_0195_3experts.png"),
]

# 标题高度和字体大小
TITLE_HEIGHT = 80
FONT_SIZE = 48
GAP_WIDTH = 30  # 图片之间的间距

# 加载所有图片
images = []
for name, path in image_paths:
    img = Image.open(path)
    print(f"{name}: {img.size}")
    images.append((name, img))

# 分析极坐标图尺寸：
# Discrete (954x1157): 极坐标图直径约 800px
# MvM (959x1313): 极坐标图直径约 700px
# MoE (2622x1191): 3个极坐标图，每个直径约 700px
#
# 为了统一，我们需要：
# - 缩小 Discrete 使其极坐标图与 MoE 中的匹配
# - MvM 的极坐标图大小与 MoE 接近

# 目标：让所有极坐标图直径约为 700px
# Discrete 需要缩放到 700/800 = 0.875
# MvM 保持原样
# MoE 保持原样

scale_factors = {
    "Discrete": 1.05,  # 稍微放大
    "MvM": 0.85,       # 缩小使极坐标图与MoE匹配
    "MoE": 1.0,
}

# 缩放图片
scaled_images = []
for name, img in images:
    scale = scale_factors[name]
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    scaled_images.append((name, img))
    print(f"{name} 缩放后: {img.size}")

# 统一高度（取最大高度）
max_height = max(img.height for _, img in scaled_images)
print(f"\n最大高度: {max_height}")

# 调整图片高度（保持宽高比）
def resize_to_height(img, target_height):
    if img.height == target_height:
        return img
    ratio = target_height / img.height
    new_width = int(img.width * ratio)
    return img.resize((new_width, target_height), Image.LANCZOS)

resized_images = [(name, resize_to_height(img, max_height)) for name, img in scaled_images]

# 计算拼接后的总宽度（包含间距）
num_gaps = len(resized_images) - 1
total_width = sum(img.width for _, img in resized_images) + num_gaps * GAP_WIDTH
print(f"总宽度: {total_width}")

# 创建新图像（加上标题高度）
result = Image.new('RGB', (total_width, max_height + TITLE_HEIGHT), (255, 255, 255))
draw = ImageDraw.Draw(result)

# 尝试加载字体
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
except:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

# 粘贴图片并添加标题
x_offset = 0
for i, (name, img) in enumerate(resized_images):
    # 粘贴图片（在标题下方）
    result.paste(img, (x_offset, TITLE_HEIGHT))

    # 绘制标题（居中）
    bbox = draw.textbbox((0, 0), name, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = x_offset + (img.width - text_width) // 2
    text_y = (TITLE_HEIGHT - (bbox[3] - bbox[1])) // 2
    draw.text((text_x, text_y), name, fill=(0, 0, 0), font=font)

    x_offset += img.width

    # 在图片之间绘制分割线（除了最后一张）
    if i < len(resized_images) - 1:
        line_x = x_offset + GAP_WIDTH // 2
        draw.line([(line_x, 10), (line_x, max_height + TITLE_HEIGHT - 10)],
                  fill=(200, 200, 200), width=2)
        x_offset += GAP_WIDTH

# 保存结果
output_path = "paper_figures/airplane_0195_comparison.png"
result.save(output_path, dpi=(300, 300))
print(f"\n拼接完成！保存至: {output_path}")
print(f"最终尺寸: {result.size}")
print(f"共 {len(images)} 张图片")
