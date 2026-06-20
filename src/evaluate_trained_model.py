import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

# ==========================
# LOAD TEST EMBEDDINGS
# ==========================

rgb = np.load(
    "remoteclip_test/rgb_test.npy"
).astype(np.float32)

sar = np.load(
    "remoteclip_test/sar_test.npy"
).astype(np.float32)

print("RGB Test Shape:", rgb.shape)
print("SAR Test Shape:", sar.shape)

# ==========================
# DEVICE
# ==========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ==========================
# MODEL
# ==========================

class ProjectionHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):

        x = self.net(x)

        x = F.normalize(
            x,
            dim=1
        )

        return x

model = ProjectionHead().to(device)

model.load_state_dict(
    torch.load(
        "remoteclip_test/contrastive_model.pth",
        map_location=device
    )
)

model.eval()

# ==========================
# ALIGN RGB TEST EMBEDDINGS
# ==========================

with torch.no_grad():

    rgb_tensor = torch.tensor(
        rgb,
        dtype=torch.float32
    ).to(device)

    aligned_rgb = model(
        rgb_tensor
    )

aligned_rgb = (
    aligned_rgb
    .cpu()
    .numpy()
    .astype(np.float32)
)

# ==========================
# NORMALIZE SAR
# ==========================

sar = sar / np.linalg.norm(
    sar,
    axis=1,
    keepdims=True
)

# ==========================
# BUILD FAISS INDEX
# ==========================

index = faiss.IndexFlatIP(512)

index.add(sar)

# ==========================
# EVALUATION
# ==========================

top1 = 0
top5 = 0
top10 = 0

N = len(aligned_rgb)

start = time.time()

for i in range(N):

    D, I = index.search(
        aligned_rgb[i:i+1],
        10
    )

    results = I[0]

    if i == results[0]:
        top1 += 1

    if i in results[:5]:
        top5 += 1

    if i in results[:10]:
        top10 += 1

end = time.time()

# ==========================
# RESULTS
# ==========================

print("\n===== RESULTS =====\n")

print(
    f"Top-1 Accuracy : {100*top1/N:.2f}%"
)

print(
    f"Top-5 Accuracy : {100*top5/N:.2f}%"
)

print(
    f"Top-10 Accuracy : {100*top10/N:.2f}%"
)

print(
    f"Queries Tested : {N}"
)

print(
    f"Average Retrieval Time : {(1000*(end-start)/N):.2f} ms"
)