import troch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, StableDiffusionDreamBoothPipeline
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import os

# Muat model
from safetensors.torch import load_file
model_path = "../models/"
model = load_file(model_path)

# Menggunkana GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Muat pipeline
pipe = StableDiffusionDreamBoothPipeline.from_pretrained("CmpVis/stable-diffusion-v1-4-original".to(device))

# Menggunakan dataset
train_dataset = load_dataset("../data/train/", split="train")

# Parameter training sederhana
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=500,
    save_total_limit=2,
)

# Konfigurasi trainer
trainer = Trainer(
    model=pipe.model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=None,
)

# Latih
trainer.train()

# Simpan
pipe.save_pretrained("../output/fine_tuned_model")
