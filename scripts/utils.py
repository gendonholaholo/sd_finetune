import os
from PIL import Image
from transformers import CLIPTokenizer
import torch
import numpy as np

def load_images_and_metadata(image_folder, metadata_folder):
    images = []
    texts = []

    for image_name in os.listdir(image_folder):
        if image_name.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder, image_name)
            img = Image.open(image_path).convert("RGB")
            images.append(img)

            metadata_path = os.path.join(metadata_folder, image_name.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt"))
            with open(metadata_path, "r") as file:
                text = file.read().strip()
            texts.append(text)

    return images, texts

def preprocess_images_and_texts(images, texts, tokenizer, max_length=77):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    processed_images = []
    for img in images:
        img = img.resize((512, 512))

        img = np.array(img)

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # Normalisasi gambar ke rentang [0, 1]
        processed_images.append(img)

    return processed_images, inputs
