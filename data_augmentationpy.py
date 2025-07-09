import os
import random
import numpy as np
from PIL import Image, ImageOps

# === Define dataset structure ===
sets = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

# Base path to preprocessed images
base_output = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_preprocessed_new'

# Output path for augmented images
augmented_output_base = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_augmented'

# === Augmentation Functions ===

# Horizontal flip
def horizontal_flip(img):
    return ImageOps.mirror(img)

# Random rotation within ±15 degrees
def random_rotation(img, angle_range=15):
    angle = random.uniform(-angle_range, angle_range)
    return img.rotate(angle, resample=Image.BILINEAR)

# Add Gaussian noise
def add_gaussian_noise(img, mean=0, std=10):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# Apply all augmentations and return list of images
def augment_image(img):
    augmented = []
    augmented.append(horizontal_flip(img))
    augmented.append(random_rotation(img))
    augmented.append(add_gaussian_noise(img))
    return augmented

# === Main Processing Loop ===
for set_name in sets:
    for cls in classes:
        input_dir = os.path.join(base_output, set_name, cls)
        output_dir = os.path.join(augmented_output_base, set_name, cls)
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, file)

                try:
                    img = Image.open(input_path).convert('L')  # grayscale
                    augmentations = augment_image(img)

                    # Save each augmented image with suffix
                    for i, aug_img in enumerate(augmentations):
                        filename = f"{os.path.splitext(file)[0]}_aug{i+1}.png"
                        aug_path = os.path.join(output_dir, filename)
                        aug_img.save(aug_path)

                except Exception as e:
                    print(f"Augmentation failed for {input_path}: {e}")
