import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split

# ==========================
# LOAD EMBEDDINGS
# ==========================

rgb = np.load(
    "remoteclip_test/rgb_embeddings.npy"
).astype(np.float32)

sar = np.load(
    "remoteclip_test/sar_embeddings.npy"
).astype(np.float32)

print("Total Pairs:", len(rgb))

# ==========================
# TRAIN / TEST SPLIT
# ==========================

indices = np.arange(len(rgb))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

rgb_train = rgb[train_idx]
sar_train = sar[train_idx]

rgb_test = rgb[test_idx]
sar_test = sar[test_idx]

np.save(
    "remoteclip_test/rgb_test.npy",
    rgb_test
)

np.save(
    "remoteclip_test/sar_test.npy",
    sar_test
)

print("Train Pairs:", len(rgb_train))
print("Test Pairs:", len(rgb_test))

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

# ==========================
# TRAIN DATA
# ==========================

rgb_tensor = torch.tensor(
    rgb_train,
    dtype=torch.float32
).to(device)

sar_tensor = torch.tensor(
    sar_train,
    dtype=torch.float32
).to(device)

# ==========================
# OPTIMIZER
# ==========================

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

epochs = 200

print("\nTraining Started...\n")

# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    rgb_proj = model(
        rgb_tensor
    )

    sar_proj = F.normalize(
        sar_tensor,
        dim=1
    )

    logits = (
        rgb_proj @ sar_proj.T
    ) / 0.07

    labels = torch.arange(
        len(rgb_proj)
    ).to(device)

    loss_rgb = F.cross_entropy(
        logits,
        labels
    )

    loss_sar = F.cross_entropy(
        logits.T,
        labels
    )

    loss = (
        loss_rgb + loss_sar
    ) / 2

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss = {loss.item():.4f}"
        )

# ==========================
# SAVE MODEL
# ==========================

torch.save(
    model.state_dict(),
    "remoteclip_test/contrastive_model.pth"
)

print("\nModel Saved!")