import numpy as np
import faiss

# Load embeddings

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

# Normalize

faiss.normalize_L2(rgb_embeddings)
faiss.normalize_L2(sar_embeddings)

# Build SAR index

dimension = sar_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(sar_embeddings)

top1 = 0
top5 = 0
top10 = 0

num_queries = len(rgb_embeddings)

for i in range(num_queries):

    query = rgb_embeddings[i].reshape(1, -1)

    D, I = index.search(query, 10)

    target_name = (
        rgb_paths[i]
        .split("/")[-1]
    )

    retrieved = [
        sar_paths[idx].split("/")[-1]
        for idx in I[0]
    ]

    if target_name == retrieved[0]:
        top1 += 1

    if target_name in retrieved[:5]:
        top5 += 1

    if target_name in retrieved[:10]:
        top10 += 1

print("\n====== RESULTS ======\n")

print(
    f"Top-1 Accuracy : {top1/num_queries*100:.2f}%"
)

print(
    f"Top-5 Accuracy : {top5/num_queries*100:.2f}%"
)

print(
    f"Top-10 Accuracy: {top10/num_queries*100:.2f}%"
)

print(
    f"Queries Tested : {num_queries}"
)