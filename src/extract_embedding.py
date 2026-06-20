from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch

print("Loading model...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model.to(device)

print("Model loaded!")

image = Image.open(
    "dataset/test_images/sample.jpg"
)

print("Image Size:", image.size)

image = image.convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state[:, 0]

print("Embedding Shape:")
print(embedding.shape)