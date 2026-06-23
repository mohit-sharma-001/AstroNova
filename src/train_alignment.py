import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

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
# DATASET
# ==========================

rgb_tensor = torch.tensor(
    rgb_train,
    dtype=torch.float32
)

sar_tensor = torch.tensor(
    sar_train,
    dtype=torch.float32
)

dataset = TensorDataset(
    rgb_tensor,
    sar_tensor
)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

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

rgb_model = ProjectionHead().to(device)
sar_model = ProjectionHead().to(device)

# ==========================
# LEARNABLE TEMPERATURE
# ==========================

logit_scale = nn.Parameter(
    torch.tensor(
        np.log(1 / 0.07),
        dtype=torch.float32,
        device=device
    )
)

# ==========================
# OPTIMIZER
# ==========================

optimizer = optim.AdamW(
    list(rgb_model.parameters()) +
    list(sar_model.parameters()) +
    [logit_scale],
    lr=1e-4,
    weight_decay=1e-4
)

epochs = 100

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

best_loss = 999999

print("\nTraining Started...\n")

# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(epochs):

    rgb_model.train()
    sar_model.train()

    epoch_loss = 0.0

    for rgb_batch, sar_batch in loader:

        rgb_batch = rgb_batch.to(device)
        sar_batch = sar_batch.to(device)

        optimizer.zero_grad()

        rgb_proj = rgb_model(rgb_batch)
        sar_proj = sar_model(sar_batch)

        scale = logit_scale.exp()

        logits = scale * (
            rgb_proj @ sar_proj.T
        )

        labels = torch.arange(
            len(rgb_proj),
            device=device
        )

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

        torch.nn.utils.clip_grad_norm_(
            list(rgb_model.parameters()) +
            list(sar_model.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    avg_loss = epoch_loss / len(loader)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            rgb_model.state_dict(),
            "remoteclip_test/rgb_model.pth"
        )

        torch.save(
            sar_model.state_dict(),
            "remoteclip_test/sar_model.pth"
        )

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Loss = {avg_loss:.4f}"
    )

print("\nTraining Complete!")
print("Best Loss:", best_loss)
print("Models Saved!")