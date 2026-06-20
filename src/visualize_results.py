from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

gallery_paths = np.load(
    "embeddings/gallery_paths.npy",
    allow_pickle=True
)

query_path = "dataset/test_images/sample.jpg"

# Replace these with your retrieved indices
retrieved = [0, 1, 2, 3, 4]

fig, axes = plt.subplots(1, 6, figsize=(18, 4))

query_img = Image.open(query_path)

axes[0].imshow(query_img)
axes[0].set_title("QUERY")
axes[0].axis("off")

for i, idx in enumerate(retrieved):
    img = Image.open(gallery_paths[idx])

    axes[i + 1].imshow(img)
    axes[i + 1].set_title(f"TOP {i+1}")
    axes[i + 1].axis("off")

plt.tight_layout()

plt.savefig(
    "results/retrieval_result.png"
)

print("Saved: results/retrieval_result.png")