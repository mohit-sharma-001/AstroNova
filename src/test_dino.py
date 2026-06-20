from transformers import AutoImageProcessor, AutoModel

print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

print("Model Loaded Successfully!")