import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
import faiss
import time

# =========================
# Load DINOv2
# =========================

print("Loading model...")

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-base"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-base"
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model.to(device)

print("Model Loaded!")

# =========================
# Load Paths
# =========================

rgb_paths = np.load(
    "embeddings/gallery_paths.npy",
    allow_pickle=True
)

sar_paths = np.load(
    "embeddings/sar_paths.npy",
    allow_pickle=True
)

ir_paths = np.load(
    "embeddings/ir_paths.npy",
    allow_pickle=True
)

all_paths = np.concatenate([
    rgb_paths,
    sar_paths,
    ir_paths
])

print("Total Images:", len(all_paths))

# =========================
# Load Unified FAISS Index
# =========================

index = faiss.read_index(
    "embeddings/unified.index"
)

print("FAISS Index Ready!")
print("Vectors:", index.ntotal)

# =========================
# CHOOSE QUERY IMAGE
# =========================

# RGB TEST
#query_path = rgb_paths[0]

# SAR TEST
#query_path = sar_paths[0]

# IR TEST
query_path = ir_paths[0]

print("\nQuery Image:")
print(query_path)

# =========================
# Extract Query Embedding
# =========================

image = Image.open(
    query_path
).convert("RGB")

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

query_embedding = outputs.last_hidden_state[:, 0]

query_embedding = (
    query_embedding
    .cpu()
    .numpy()
    .astype("float32")
)

faiss.normalize_L2(query_embedding)

# =========================
# Search
# =========================

start = time.time()

D, I = index.search(
    query_embedding,
    5
)

end = time.time()

print("\nTop 5 Results\n")

for rank, idx in enumerate(I[0]):

    path = all_paths[idx]

    if "RGB_Gallery" in path:
        modality = "RGB"
    elif "SAR_Gallery" in path:
        modality = "SAR"
    elif "IR_Gallery" in path:
        modality = "IR"
    else:
        modality = "UNKNOWN"

    print(
        f"{rank+1}. [{modality}] {path}"
    )

print("\nModalities Returned:")

for idx in I[0]:

    path = all_paths[idx]

    if "RGB_Gallery" in path:
        print("RGB")
    elif "SAR_Gallery" in path:
        print("SAR")
    elif "IR_Gallery" in path:
        print("IR")

print(
    f"\nRetrieval Time: {(end-start)*1000:.2f} ms"
)

   