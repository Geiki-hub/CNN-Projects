import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allow processing very large images

# Set local paths
input_wsi_path = r'E:\2. PREPROCESSED DATASET\P48-18 1 A200482 H&E.jpeg'  # Update to your image file path
output_folder_path = r'E:\2. PREPROCESSED DATASET'  # Update to your output folder path
output_filename = "P48-18 1 A200482 H&E.jpeg (N).png"  # Output file name


# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

def normalize_image(input_path, output_folder, output_filename):
    try:
        # Load image using PIL
        img = Image.open(input_path).convert("RGB")  # Ensure RGB mode

        # Convert image to NumPy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize pixel values to range [0, 255]
        img_min, img_max = img_array.min(), img_array.max()
        normalized_img = ((img_array - img_min) / (img_max - img_min)) * 255.0
        normalized_img = normalized_img.astype(np.uint8)  # Convert back to uint8

        # Convert back to PIL Image
        normalized_img = Image.fromarray(normalized_img)

        # Save the normalized image
        output_path = os.path.join(output_folder, output_filename)
        normalized_img.save(output_path)

        print(f"Image normalized and saved at {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    normalize_image(input_wsi_path, output_folder_path, output_filename)