import os
from PIL import Image

def load_and_resize(path, size=(224, 224)):
    img = Image.open(path).convert('RGB')
    return img.resize(size)

def rgb_to_grayscale(img):
    width, height = img.size
    gray_img = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            gray_img.putpixel((x, y), gray)

    return gray_img

def histogram_equalization(img):
    pixels = list(img.getdata())
    hist = [0] * 256
    for val in pixels:
        hist[val] += 1

    cdf = []
    cum_sum = 0
    for h in hist:
        cum_sum += h
        cdf.append(cum_sum)

    cdf_min = next(v for v in cdf if v > 0)
    total_pixels = len(pixels)

    equalized = [int((cdf[p] - cdf_min) * 255 / (total_pixels - cdf_min)) for p in pixels]
    new_img = Image.new('L', img.size)
    new_img.putdata(equalized)
    return new_img

def contrast_stretch(img):
    pixels = list(img.getdata())
    min_val = min(pixels)
    max_val = max(pixels)

    if max_val == min_val:
        stretched = [0] * len(pixels)
    else:
        stretched = [int((p - min_val) * 255 / (max_val - min_val)) for p in pixels]

    new_img = Image.new('L', img.size)
    new_img.putdata(stretched)
    return new_img

def preprocess_image(path):
    img = load_and_resize(path)
    gray = rgb_to_grayscale(img)
    eq = histogram_equalization(gray)
    contrast = contrast_stretch(eq)
    return contrast

def save_image(img, output_path):
    img.save(output_path)


base_input = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive\chest_xray'
base_output = r'C:\Users\user\OneDrive - University of Hertfordshire\project\archive_preprocessed'

sets = ['train', 'val', 'test']
classes = ['NORMAL', 'PNEUMONIA']

for set_name in sets:
    for cls in classes:
        input_dir = os.path.join(base_input, set_name, cls)
        output_dir = os.path.join(base_output, set_name, cls)
        os.makedirs(output_dir, exist_ok=True)

        for file in os.listdir(input_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, file)
                output_path = os.path.join(output_dir, file)

                try:
                    processed = preprocess_image(input_path)
                    save_image(processed, output_path)
                    print(f"Processed: {output_path}")
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")
