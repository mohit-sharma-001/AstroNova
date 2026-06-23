import os
import numpy as np
import torch
import open_clip

from PIL import Image

print("Loading RemoteCLIP...")

# =========================
# MODEL
# =========================

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = model.to(device)
model.eval()

print("RemoteCLIP Ready!")

# =========================
# FIND ALL REGIONS
# =========================

dataset_root = "dataset"

rgb_paths_dict = {}
sar_paths_dict = {}

regions = sorted([
    d for d in os.listdir(dataset_root)
    if d.startswith("r_")
])

print(f"Found {len(regions)} regions")

for region in regions:

    region_path = os.path.join(
        dataset_root,
        region
    )

    region_num = region.split("_")[1]

    sar_folder = os.path.join(
        region_path,
        f"s1_{region_num}"
    )

    rgb_folder = os.path.join(
        region_path,
        f"s2_{region_num}"
    )

    if (
        not os.path.exists(sar_folder)
        or
        not os.path.exists(rgb_folder)
    ):
        continue

    rgb_files = sorted([
        f for f in os.listdir(rgb_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    sar_files = sorted([
        f for f in os.listdir(sar_folder)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    common = sorted(
        list(
            set(rgb_files).intersection(
                set(sar_files)
            )
        )
    )

    print(
        f"{region} -> {len(common)} pairs"
    )

    for fname in common:

        unique_name = (
            f"{region}_{fname}"
        )

        rgb_paths_dict[
            unique_name
        ] = os.path.join(
            rgb_folder,
            fname
        )

        sar_paths_dict[
            unique_name
        ] = os.path.join(
            sar_folder,
            fname
        )

common_files = sorted(
    rgb_paths_dict.keys()
)

print(
    f"\nTotal Paired Images: {len(common_files)}"
)

# =========================
# STORAGE
# =========================

rgb_embeddings = []
sar_embeddings = []

rgb_paths = []
sar_paths = []

# =========================
# EXTRACT EMBEDDINGS
# =========================

for i, filename in enumerate(common_files):

    try:

        rgb_path = rgb_paths_dict[
            filename
        ]

        sar_path = sar_paths_dict[
            filename
        ]

        rgb_img = Image.open(
            rgb_path
        ).convert("RGB")

        sar_img = Image.open(
            sar_path
        ).convert("RGB")

        rgb_tensor = preprocess(
            rgb_img
        ).unsqueeze(0).to(device)

        sar_tensor = preprocess(
            sar_img
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            rgb_feat = model.encode_image(
                rgb_tensor
            )

            sar_feat = model.encode_image(
                sar_tensor
            )

        rgb_feat = (
            rgb_feat
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        sar_feat = (
            sar_feat
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        rgb_embeddings.append(
            rgb_feat
        )

        sar_embeddings.append(
            sar_feat
        )

        rgb_paths.append(
            rgb_path
        )

        sar_paths.append(
            sar_path
        )

        if (i + 1) % 100 == 0:

            print(
                f"Processed {i+1}/{len(common_files)}"
            )

    except Exception as e:

        print(
            f"Skipping {filename}: {e}"
        )

# =========================
# CONVERT TO NUMPY
# =========================

rgb_embeddings = np.array(
    rgb_embeddings,
    dtype=np.float32
)

sar_embeddings = np.array(
    sar_embeddings,
    dtype=np.float32
)

# =========================
# SAVE
# =========================

os.makedirs(
    "remoteclip_test",
    exist_ok=True
)

np.save(
    "remoteclip_test/rgb_embeddings.npy",
    rgb_embeddings
)

np.save(
    "remoteclip_test/sar_embeddings.npy",
    sar_embeddings
)

np.save(
    "remoteclip_test/rgb_paths.npy",
    np.array(rgb_paths)
)

np.save(
    "remoteclip_test/sar_paths.npy",
    np.array(sar_paths)
)

# =========================
# REPORT
# =========================

print("\n===== DONE =====")

print(
    "RGB Shape:",
    rgb_embeddings.shape
)

print(
    "SAR Shape:",
    sar_embeddings.shape
)