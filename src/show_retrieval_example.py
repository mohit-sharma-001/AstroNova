import numpy as np
import faiss
import matplotlib.pyplot as plt
from PIL import Image

# ==========================
# LOAD EMBEDDINGS
# ==========================

rgb_embeddings = np.load(
    "remoteclip_test/rgb_embeddings.npy"
).astype("float32")

sar_embeddings = np.load(
    "remoteclip_test/sar_embeddings.npy"
).astype("float32")

rgb_paths = np.load(
    "remoteclip_test/rgb_paths.npy",
    allow_pickle=True
)

sar_paths = np.load(
    "remoteclip_test/sar_paths.npy",
    allow_pickle=True
)

# ==========================
# NORMALIZE
# ==========================

faiss.normalize_L2(rgb_embeddings)
faiss.normalize_L2(sar_embeddings)

# ==========================
# BUILD RGB INDEX
# ==========================

index = faiss.IndexFlatIP(
    rgb_embeddings.shape[1]
)

index.add(rgb_embeddings)

# ==========================
# CHOOSE QUERY
# ==========================

query_idx = 0

query_embedding = sar_embeddings[
    query_idx
].reshape(1, -1)

# ==========================
# SEARCH TOP-10 RGB
# ==========================

D, I = index.search(
    query_embedding,
    10
)

# ==========================
# PRINT RESULTS
# ==========================

print("\n===== QUERY SAR =====\n")

print(
    sar_paths[query_idx]
)

print("\n===== GROUND TRUTH RGB =====\n")

print(
    rgb_paths[query_idx]
)

print("\n===== RETRIEVED RGB RESULTS =====\n")

for rank, idx in enumerate(I[0]):

    print(
        f"Top-{rank+1}: {rgb_paths[idx]}"
    )

# ==========================
# LOAD QUERY + GT
# ==========================

query_sar = Image.open(
    sar_paths[query_idx]
).convert("RGB")

ground_truth_rgb = Image.open(
    rgb_paths[query_idx]
).convert("RGB")

# ==========================
# CREATE FIGURE
# ==========================

fig, axes = plt.subplots(
    1,
    12,
    figsize=(36, 4)
)

# ==========================
# SAR QUERY
# ==========================

axes[0].imshow(query_sar)
axes[0].set_title(
    "SAR Query",
    fontsize=12
)
axes[0].axis("off")

# ==========================
# GROUND TRUTH
# ==========================

axes[1].imshow(ground_truth_rgb)
axes[1].set_title(
    "Ground Truth",
    fontsize=12
)
axes[1].axis("off")

# ==========================
# TOP-10 RESULTS
# ==========================

for rank, idx in enumerate(I[0]):

    img = Image.open(
        rgb_paths[idx]
    ).convert("RGB")

    axes[rank + 2].imshow(img)

    axes[rank + 2].set_title(
        f"Top-{rank+1}",
        fontsize=10
    )

    axes[rank + 2].axis("off")

# ==========================
# SAVE
# ==========================

plt.tight_layout()

plt.savefig(
    "retrieval_result_top10.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved retrieval_result_top10.png"
)