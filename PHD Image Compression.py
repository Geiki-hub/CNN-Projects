from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

input_path = r"D:\TANG GEI KI\PHD LOSSLESS COMPRESSED\P3019-19-2 B718012.png"
output_path = r"D:\TANG GEI KI\PHD LOSSLESS COMPRESSED\P3019-19-2 B718012 (2).png"

img = Image.open(input_path).convert("RGB")

w, h = img.size
img_half = img.resize((w // 2, h // 2), Image.LANCZOS)

img_half.save(
    output_path,
    format="PNG",          # lossless storage of resized image
    optimize=True
)

print("✅ Image downsampled by 2× using Lanczos (high-quality)")
