import os
from PIL import Image
import matplotlib.pyplot as plt

# Base directory of preprocessed dataset
base_dir = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_preprocessed_new'
sets = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

# Loop through each set
for set_name in sets:
    for cls in classes:
        folder = os.path.join(base_dir, set_name, cls)
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if images:
            img_path = os.path.join(folder, images[0])
            img = Image.open(img_path)
            plt.imshow(img, cmap='gray')
            plt.title(f"{set_name}/{cls}")
            plt.axis('off')
            plt.show()
