import os
import random
import numpy as np
from PIL import Image, ImageOps

# Set categories and paths
classes = ['NORMAL', 'PNEUMONIA']
input_root = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_preprocessed\train'
output_root = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_augmented_combined\train'

# Create augmentation functions
def horizontal_flip(img):
    return ImageOps.mirror(img)

def random_rotation(img, angle_range=15):
    angle = random.uniform(-angle_range, angle_range)
    return img.rotate(angle, resample=Image.BILINEAR)

def add_gaussian_noise(img, mean=0, std=10):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_augmentations(img):
    return [
        horizontal_flip(img),
        random_rotation(img),
        add_gaussian_noise(img)
    ]

# Loop through each class (NORMAL, PNEUMONIA)
for cls in classes:
    input_dir = os.path.join(input_root, cls)
    output_dir = os.path.join(output_root, cls)
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            name_without_ext = os.path.splitext(filename)[0]

            try:
                # Open and convert to grayscale
                img = Image.open(input_path).convert('L')

                # Save original image
                original_save_path = os.path.join(output_dir, f"{name_without_ext}.png")
                img.save(original_save_path)

                # Save augmented versions
                augmented_images = apply_augmentations(img)
                for i, aug_img in enumerate(augmented_images):
                    aug_name = f"{name_without_ext}_aug{i+1}.png"
                    aug_path = os.path.join(output_dir, aug_name)
                    aug_img.save(aug_path)

            except Exception as e:
                print(f"[ERROR] Failed processing {input_path}: {e}")
