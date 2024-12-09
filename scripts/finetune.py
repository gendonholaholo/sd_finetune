import torch
from diffusers import StableDiffusionPipeline
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from utils import load_images_and_metadata, preprocess_images_and_texts
from transformers import CLIPTokenizer
import os

model_path = "E:/Developer/Program/AI/stable-diffusion-finetune/models/sd-v1-4.ckpt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(model_path, map_location=device)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",  
    revision="fp16", 
    torch_dtype=torch.float16
)

pipe.to(device)

pipe.unet.load_state_dict(checkpoint["state_dict"], strict=False)
pipe.vae.load_state_dict(checkpoint["state_dict"], strict=False)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image_folder = "E:/Developer/Program/AI/stable-diffusion-finetune/data/train/images"
metadata_folder = "E:/Developer/Program/AI/stable-diffusion-finetune/data/train/metadata"
images, texts = load_images_and_metadata(image_folder, metadata_folder)

processed_images, inputs = preprocess_images_and_texts(images, texts, tokenizer)

train_data = Dataset.from_dict({
    'pixel_values': processed_images,
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask']
})

training_args = TrainingArguments(
    output_dir="./output",  
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=pipe.unet,  
    args=training_args,
    train_dataset=train_data,
    data_collator=None,  
)

trainer.train()

pipe.save_pretrained("E:/Developer/Program/AI/stable-diffusion-finetune/output/fine_tuned_model")
