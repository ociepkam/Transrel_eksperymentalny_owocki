import requests
from PIL import Image
import numpy as np
import os

# Pobierz obrazek lub wczytaj lokalnie
img = Image.open("image.png")  # jeśli masz plik lokalnie

# Zamiana na RGBA
if img.mode != 'RGBA':
    img = img.convert('RGBA')

width, height = img.size

# Współrzędne (proporcje) dla każdego obiektu
fruit_boxes = [
    [0.08, 0.05, 0.22, 0.40],   # banan
    [0.23, 0.12, 0.41, 0.45],   # cytryna
    [0.41, 0.08, 0.54, 0.40],   # jabłko
    [0.54, 0.10, 0.66, 0.40],   # truskawka
    [0.66, 0.12, 0.78, 0.40],   # gruszka
    [0.78, 0.06, 0.89, 0.41],   # winogrona
]
vegetable_boxes = [
    [0.06, 0.60, 0.12, 0.90],   # chili
    [0.12, 0.59, 0.25, 0.90],   # papryka
    [0.25, 0.56, 0.41, 0.90],   # cukinia
    [0.41, 0.60, 0.52, 0.90],   # ziemniak
    [0.52, 0.53, 0.69, 0.90],   # brokuły
    [0.69, 0.65, 0.825, 0.90],   # pomidor
]

def extract_object(img, box, target_size=256):
    left = int(box[0] * width)
    top = int(box[1] * height)
    right = int(box[2] * width)
    bottom = int(box[3] * height)
    cropped = img.crop((left, top, right, bottom))
    square_img = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    cropped_width, cropped_height = cropped.size
    if cropped_width > cropped_height:
        new_width = target_size
        new_height = int((cropped_height / cropped_width) * target_size)
    else:
        new_height = target_size
        new_width = int((cropped_width / cropped_height) * target_size)
    resized = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    square_img.paste(resized, (x_offset, y_offset), resized)
    data = np.array(square_img)
    white_threshold = 240
    white_mask = (data[:, :, 0] > white_threshold) & \
                 (data[:, :, 1] > white_threshold) & \
                 (data[:, :, 2] > white_threshold)
    data[white_mask] = [0, 0, 0, 0]
    return Image.fromarray(data, 'RGBA')

os.makedirs('extracted_objects', exist_ok=True)

for i, box in enumerate(fruit_boxes, 1):
    fruit_img = extract_object(img, box)
    fruit_img.save(f'extracted_objects/fruit_{i}.png')

for i, box in enumerate(vegetable_boxes, 1):
    vegetable_img = extract_object(img, box)
    vegetable_img.save(f'extracted_objects/vegetable_{i}.png')

print("Wszystkie pliki zostały utworzone w folderze 'extracted_objects'.")