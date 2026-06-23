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
# NORMALIZE INPUT EMBEDDINGS
# ==========================

rgb = rgb / np.linalg.norm(
    rgb,
    axis=1,
    keepdims=True
)

sar = sar / np.linalg.norm(
    sar,
    axis=1,
    keepdims=True
)

# ==========================
# DEVICE
# ==========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

# ==========================
# MODEL
# ==========================

class ProjectionHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)

        )

    def forward(self, x):

        x = self.net(x)

        return F.normalize(
            x,
            dim=1
        )
# ==========================
# LOAD MODELS
# ==========================

rgb_model = ProjectionHead().to(device)
sar_model = ProjectionHead().to(device)

rgb_model.load_state_dict(
    torch.load(
        "remoteclip_test/rgb_model.pth",
        map_location=device
    )
)

sar_model.load_state_dict(
    torch.load(
        "remoteclip_test/sar_model.pth",
        map_location=device
    )
)

rgb_model.eval()
sar_model.eval()

print("Models Loaded!")

# ==========================
# PROJECT TEST EMBEDDINGS
# ==========================

with torch.no_grad():

    rgb_tensor = torch.tensor(
        rgb,
        dtype=torch.float32
    ).to(device)

    sar_tensor = torch.tensor(
        sar,
        dtype=torch.float32
    ).to(device)

    rgb_proj = rgb_model(
        rgb_tensor
    )

    sar_proj = sar_model(
        sar_tensor
    )

rgb_proj = (
    rgb_proj
    .cpu()
    .numpy()
    .astype(np.float32)
)

sar_proj = (
    sar_proj
    .cpu()
    .numpy()
    .astype(np.float32)
)

# ==========================
# BUILD FAISS INDEX
# ==========================

dimension = sar_proj.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(sar_proj)

print("FAISS Index Ready!")

# ==========================
# EVALUATION
# ==========================

top1 = 0
top5 = 0
top10 = 0

N = len(rgb_proj)

start = time.time()

for i in range(N):

    D, I = index.search(
        rgb_proj[i:i+1],
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
    f"Average Retrieval Time : {(1000*(end-start)/N):.4f} ms"
)