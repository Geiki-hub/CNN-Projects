import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from patchify import patchify

Image.MAX_IMAGE_PIXELS = None

# Set the image path (modify this to your local path)
img_path = r"C:\TANG GEI KI WSI SPLIT\TESTING (10%)\MYC-\P2363-16 B616878 H&E (N).png"  # Replace with the actual path

# Check if the file exists
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image file not found: {img_path}")

# Load the image using PIL
img = Image.open(img_path)

# Display image properties
width, height = img.size
mode = img.mode
print(f"Image loaded successfully. Dimensions: {width}x{height}, Mode: {mode}")

# Folder path to save the patches
output_folder = r"D:\TANG GEI KI\MYC-\P2363-16 B616878 H&E (N).png"  # Modify as needed

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Desired patch size
patch_size = 256  # Set this based on memory constraints

# Convert image to NumPy array
img_array = np.array(img)

# Patchify the image into 256x256 non-overlapping patches
patches = patchify(img_array, (patch_size, patch_size, 3), step=patch_size)

# Save each patch
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]  # Extract the patch
        patch_img = Image.fromarray(patch)  # Convert NumPy array to PIL image

        # Construct the output filename
        output_filename = os.path.join(output_folder, f"P2363-16 B616878_patch_{i}_{j}.png")

        # Save as PNG
        patch_img.save(output_filename, "PNG")

print("Image patching completed. Patches saved in:", output_folder)
