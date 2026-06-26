import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import csv
import matplotlib.pyplot as plt


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

queries_data = []
precisions_5 = []
precisions_10 = []
recalls_5 = []
recalls_10 = []
f1s_5 = []
f1s_10 = []
aps = []

for i in range(N):
    D, I = index.search(
        rgb_proj[i:i+1],
        10
    )
    results = I[0]

    # Ground truth index for query i is i
    if i in results:
        r = np.where(results == i)[0][0]
        rank = r + 1
        ap = 1.0 / rank
    else:
        rank = -1
        ap = 0.0
    aps.append(ap)

    # Top-K accuracy checks
    if i == results[0]:
        top1 += 1

    if i in results[:5]:
        top5 += 1

    if i in results[:10]:
        top10 += 1

    # Precision@5 / Recall@5 / F1@5
    in_5 = i in results[:5]
    p_5 = 1.0 / 5.0 if in_5 else 0.0
    r_5 = 1.0 if in_5 else 0.0
    f1_5 = 2.0 * (p_5 * r_5) / (p_5 + r_5) if in_5 else 0.0
    precisions_5.append(p_5)
    recalls_5.append(r_5)
    f1s_5.append(f1_5)

    # Precision@10 / Recall@10 / F1@10
    in_10 = i in results[:10]
    p_10 = 1.0 / 10.0 if in_10 else 0.0
    r_10 = 1.0 if in_10 else 0.0
    f1_10 = 2.0 * (p_10 * r_10) / (p_10 + r_10) if in_10 else 0.0
    precisions_10.append(p_10)
    recalls_10.append(r_10)
    f1s_10.append(f1_10)

    # Save per-query metrics
    queries_data.append({
        "query_index": i,
        "gt_rank": rank,
        "ap": ap,
        "precision_5": p_5,
        "precision_10": p_10,
        "recall_5": r_5,
        "recall_10": r_10,
        "f1_5": f1_5,
        "f1_10": f1_10
    })

end = time.time()

# ==========================
# CALCULATE AVERAGE METRICS
# ==========================

mean_p5 = np.mean(precisions_5)
mean_p10 = np.mean(precisions_10)
mean_r5 = np.mean(recalls_5)
mean_r10 = np.mean(recalls_10)
mean_f1_5 = np.mean(f1s_5)
mean_f1_10 = np.mean(f1s_10)
mean_ap = np.mean(aps)

# ==========================
# SAVE TO CSV
# ==========================

os.makedirs("results", exist_ok=True)
csv_path = "results/evaluation_metrics.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "query_index", "gt_rank", "ap", 
        "precision_5", "precision_10", 
        "recall_5", "recall_10", 
        "f1_5", "f1_10"
    ])
    writer.writeheader()
    writer.writerows(queries_data)

# ==========================
# GENERATE PLOTS
# ==========================

# 1. Precision, Recall, F1 Chart
metrics_names = ["Precision", "Recall", "F1-Score"]
values_k5 = [mean_p5 * 100, mean_r5 * 100, mean_f1_5 * 100]
values_k10 = [mean_p10 * 100, mean_r10 * 100, mean_f1_10 * 100]

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values_k5, width, label="K = 5", color="#1f77b4")
plt.bar(x + width/2, values_k10, width, label="K = 10", color="#ff7f0e")

plt.ylabel("Score (%)")
plt.title("AstroNova Evaluation: Precision, Recall & F1-Score (K=5 vs K=10)")
plt.xticks(x, metrics_names)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

for idx_val, v in enumerate(values_k5):
    plt.text(idx_val - width/2, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
for idx_val, v in enumerate(values_k10):
    plt.text(idx_val + width/2, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("precision_recall_f1.png", dpi=300)
plt.close()

# 2. mAP Distribution Graph
plt.figure(figsize=(10, 6))
plt.hist(aps, bins=11, range=(0, 1.1), edgecolor='black', color='#2ca02c', rwidth=0.8, align='left')
plt.xlabel("Average Precision (AP)")
plt.ylabel("Number of Queries")
plt.title(f"Distribution of Average Precision (mAP = {mean_ap*100:.2f}%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()
plt.savefig("map_graph.png", dpi=300)
plt.close()

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

print("\n===== EXTENDED RESULTS =====\n")
print(f"Mean Average Precision (mAP) : {100*mean_ap:.2f}%")
print("-" * 40)
print(f"Metrics @ K = 5:")
print(f"  Precision @ 5 : {100*mean_p5:.2f}%")
print(f"  Recall @ 5    : {100*mean_r5:.2f}%")
print(f"  F1-Score @ 5  : {100*mean_f1_5:.2f}%")
print("-" * 40)
print(f"Metrics @ K = 10:")
print(f"  Precision @ 10: {100*mean_p10:.2f}%")
print(f"  Recall @ 10   : {100*mean_r10:.2f}%")
print(f"  F1-Score @ 10 : {100*mean_f1_10:.2f}%")
print("=" * 40)
print(f"Results saved to: {csv_path}")
print("Graphs saved to: precision_recall_f1.png, map_graph.png")