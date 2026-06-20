import numpy as np
import faiss

print("Loading embeddings...")

rgb = np.load("embeddings/gallery_embeddings.npy")
sar = np.load("embeddings/sar_embeddings.npy")
ir = np.load("embeddings/ir_embeddings.npy")

all_embeddings = np.vstack([
    rgb,
    sar,
    ir
]).astype("float32")

print("Total embeddings:", all_embeddings.shape)

faiss.normalize_L2(all_embeddings)

dimension = all_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(all_embeddings)

faiss.write_index(
    index,
    "embeddings/unified.index"
)

print("Unified index created!")
print("Vectors:", index.ntotal)