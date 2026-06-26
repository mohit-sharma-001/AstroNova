# AstroNova: Cross-Modal Remote Sensing Image Retrieval

AstroNova is a specialized machine learning backend for **cross-modal and multi-modal satellite image retrieval**. It aligns optical (RGB) and Synthetic Aperture Radar (SAR) imagery, as well as Near-Infrared (IR) data, into a shared vector embedding space. 

By training custom contrastive projection heads on top of the **RemoteCLIP** foundation model, AstroNova enables ultra-fast, cross-modal searches (e.g., querying with an optical image to find its radar counterpart) in **under 0.2 milliseconds**.

---

## 🚀 Key Features

* **Cross-Modal Subspace Alignment**: Maps completely different wavelengths (RGB optical reflection vs. SAR microwave radar scattering) into a single aligned vector space.
* **Symmetric Contrastive Learning**: Projects embeddings using custom 3-layer MLP projection heads trained with a learnable temperature InfoNCE loss.
* **Unseen Domain Generalization**: Evaluation split proves that the alignment generalized effectively to brand-new, unseen coordinates on Earth.
* **Sub-Millisecond Vector Search**: Utilizes **FAISS (Facebook AI Similarity Search)** to query thousands of satellite images with sub-millisecond latency.
* **Dual Pipeline Pipelines**:
  1. *AstroNova Aligned Projections (RemoteCLIP)* for high-accuracy cross-modal matching.
  2. *Unified Vector Database (DINOv2)* for vision-only multi-modal indexing.

---

## 📊 Performance & Hackathon Results

Evaluating AstroNova on the **1,928 pair test split** (unseen during training) shows a massive relative performance boost over the baseline pre-trained RemoteCLIP model:

| Metric | True Baseline (Raw RemoteCLIP) | AstroNova (Projected) | Relative Performance Gain |
| :--- | :---: | :---: | :---: |
| **Mean Average Precision (mAP)** | 0.74% | **29.35%** | **+3,866%** (39x increase) |
| **Top-1 Accuracy** | 0.31% | **17.74%** | **+5,622%** (57x increase) |
| **Top-5 Accuracy (Recall@5)** | 1.04% | **45.28%** | **+4,253%** (43x increase) |
| **Top-10 Accuracy (Recall@10)** | 2.28% | **59.28%** | **+2,499%** (26x increase) |
| **Precision@5** | 0.21% | **9.06%** | **+4,214%** (43x increase) |
| **F1-Score@5** | 0.35% | **15.09%** | **+4,211%** (43x increase) |

* **Average Query Latency**: **0.1503 ms** (CPU-only FAISS query).
* **Training Convergence**: Contrastive training loss steadily converged from **4.19** down to **0.24** over 100 epochs.

---

## 📁 Repository Structure

```
AstroNova/
│
├── dataset/                    # Satellite imagery organized by region (r_xxx)
│   ├── RGB_Gallery/            # Extracted visible-light optical gallery
│   ├── SAR_Gallery/            # Extracted Synthetic Aperture Radar gallery
│   └── IR_Gallery/             # Near-Infrared heatmap images (colormap applied)
│
├── embeddings/                 # Extracted DINOv2 embeddings and unified index
│
├── remoteclip_test/            # Extracted RemoteCLIP embeddings and trained model weights
│
├── results/                    # Output logs and visualizations
│
├── src/                        # Source Code
│   ├── train_alignment.py      # Core training script for projection heads
│   ├── evaluate_trained_model.py # Evaluates custom metrics and prints reports
│   ├── retrieve.py             # DINOv2 real-time query interface
│   ├── show_retrieval_example.py # Plots visual query/match demonstrations
│   └── experimental/
│       └── download_50_pairs.py # Helper script to fetch and prepare SEN12MS-CR dataset
│
├── requirements.txt            # Python dependencies
└── README.md                   # This project manual
```

---

## 🛠️ Setup & Installation

### 1. Create a Virtual Environment
Navigate to the project root directory and set up a virtual environment:
```powershell
python -m venv .venv
```

### 2. Activate the Environment
* **On Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **On Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🏃 How to Run the Code

### 1. Dataset Initialization (50 Pair Prototype)
If you want to fetch and preprocess prototype pairs from Hugging Face (`sen12mscr` dataset):
```powershell
python src/experimental/download_50_pairs.py
```

### 2. Extract RemoteCLIP Embeddings
Walks through the region folders and extracts raw embeddings:
```powershell
python src/extract_remoteclip_embeddings.py
```

### 3. Train the Projection Heads
Trains the alignment layers to bridge RGB and SAR modalities:
```powershell
python src/train_alignment.py
```

### 4. Evaluate the Trained Model
Computes detailed IR metrics (Recall, Precision, F1, mAP), saves outputs to `results/evaluation_metrics.csv`, and outputs performance charts (`precision_recall_f1.png`, `map_graph.png`):
```powershell
python src/evaluate_trained_model.py
```

### 5. Run a Real-Time Query
Queries the DINOv2 unified index with an image:
```powershell
python src/retrieve.py
```

---

## 🖼️ Visualizations & Plots
* **`loss_curve.png`**: Visualizes convergence of the projection model.
* **`baseline_vs_astronova_extended.png`**: 2-panel chart comparing baseline vs. AstroNova across all metrics.
* **`precision_recall_f1.png`**: Grouped bar chart comparison at $K=5$ and $K=10$.
* **`map_graph.png`**: Histogram showing AP distribution across test queries.
* **`retrieval_result_top10.png`**: Visual demonstration showing a SAR radar query matched with its correct ground-truth optical image.
