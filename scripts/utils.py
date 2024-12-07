import os
from PIL import Image
from transformers import CLIPTokenizer
import torch

def load_images_and_metadata(image_folder, metadata_folder):
    images = []
    texts = []

    # Membaca gambar
    for image_name in os.listdir(image_folder):
        image _path.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(image_path).convert("RGB")
            images.append(img)

            # Membaca metadata
            metadata_path = os.path.join(metadata_folder, image_name.replace(".jpg", ".txt"))
            with open(metadata_path, "r") as file:
                text = file.read().strip()
            texts.append(text)

    return images, text

def preprocess_images_and_texts(images, texts, tokenizer, max_length=77):
    # Mentokenisasi teks
    input = tokenizer(texts, padding=True, truncation=True,max_length=max_length, return_tensors="pt")
    
    processed_images = []
    for img in images:
        img = img.resize((512, 512)) # Untuk pengingat, ukurannya bebas, tapi standarnya segitu
        img = torch.tensor(img).permute(2,0,1).float()/255.0
        processed_images.append(img)

    
    return processed_images, inputs
