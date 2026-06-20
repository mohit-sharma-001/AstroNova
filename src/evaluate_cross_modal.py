import numpy as np
import faiss
import os

# ==========================
# LOAD DATA
# ==========================

rgb_embeddings = np.load(
    "embeddings/gallery_embeddings.npy"
).astype("float32")

sar_embeddings = np.load(
    "embeddings/sar_embeddings.npy"
).astype("float32")

rgb_paths = np.load(
    "embeddings/gallery_paths.npy",
    allow_pickle=True
)

sar_paths = np.load(
    "embeddings/sar_paths.npy",
    allow_pickle=True
)

# ==========================
# BUILD SAR INDEX
# ==========================

dimension = sar_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(sar_embeddings)

print("SAR Index Ready")
print("SAR Images:", len(sar_paths))

# ==========================
# RGB -> SAR EVALUATION
# ==========================

top1_correct = 0
top5_correct = 0

total_queries = min(100, len(rgb_paths))

for i in range(total_queries):

    query_embedding = rgb_embeddings[i].reshape(1, -1)

    D, I = index.search(
        query_embedding,
        5
    )

    rgb_id = os.path.basename(
        rgb_paths[i]
    )

    retrieved = [
        os.path.basename(
            sar_paths[idx]
        )
        for idx in I[0]
    ]

    # Top 1
    if rgb_id == retrieved[0]:
        top1_correct += 1

    # Top 5
    if rgb_id in retrieved:
        top5_correct += 1

# ==========================
# RESULTS
# ==========================

top1_accuracy = (
    top1_correct / total_queries
) * 100

top5_accuracy = (
    top5_correct / total_queries
) * 100

print("\n========== RESULTS ==========")

print(
    f"Top-1 Accuracy: {top1_accuracy:.2f}%"
)

print(
    f"Top-5 Accuracy: {top5_accuracy:.2f}%"
)

print(
    f"Queries Tested: {total_queries}"
)