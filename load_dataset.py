import os
from PIL import Image
import matplotlib.pyplot as plt

# Base directory of dataset
base_dir = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive\chest_xray'
sets = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

# Track grand total
grand_total = 0

# Loop through each set
for set_name in sets:
    print(f"\n=== {set_name.upper()} SET ===")
    total_images = 0

    for cls in classes:
        folder = os.path.join(base_dir, set_name, cls)
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        print(f"{cls}: {count} images")

        # Show only 1 image from each class
        if images:
            img_path = os.path.join(folder, images[0])
            img = Image.open(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(f"{set_name}/{cls} Example")
            plt.axis('off')
            plt.show()

    print(f"Total images in {set_name}: {total_images}")
    grand_total += total_images

# Grand total
print(f"\n=== GRAND TOTAL: {grand_total} images ===")
