from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import os
import numpy as np

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

gallery_folder = "dataset/RGB_Gallery"

embeddings = []
image_paths = []

for filename in os.listdir(gallery_folder):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        path = os.path.join(gallery_folder, filename)

        image = Image.open(path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0]

        embedding = (
            embedding
            .cpu()
            .numpy()
            .flatten()
        )

        embeddings.append(embedding)
        image_paths.append(path)

        print(f"Processed: {filename}")

embeddings = np.array(embeddings)

np.save(
    "embeddings/gallery_embeddings.npy",
    embeddings
)

np.save(
    "embeddings/gallery_paths.npy",
    image_paths
)

print("\nDone!")
print("Embeddings Shape:", embeddings.shape)