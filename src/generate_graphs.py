import matplotlib.pyplot as plt
import numpy as np

# ==========================
# GRAPH 1
# Final Accuracy
# ==========================

metrics = ["Top-1", "Top-5", "Top-10"]
accuracy = [17.38, 44.14, 58.97]

plt.figure(figsize=(8,5))
plt.bar(metrics, accuracy)
plt.title("AstroNova Retrieval Accuracy")
plt.ylabel("Accuracy (%)")
plt.savefig("accuracy_graph.png", dpi=300)
plt.close()

# ==========================
# GRAPH 2
# Baseline vs AstroNova
# ==========================

baseline = [5.39, 18.00, 27.59]
astronova = [17.38, 44.14, 58.97]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x-width/2, baseline, width, label="RemoteCLIP")
plt.bar(x+width/2, astronova, width, label="AstroNova")

plt.xticks(x, metrics)
plt.ylabel("Accuracy (%)")
plt.title("Baseline vs AstroNova")
plt.legend()

plt.savefig(
    "comparison_graph.png",
    dpi=300
)

plt.close()

# ==========================
# GRAPH 3
# Improvement
# ==========================

improvement = [
    ((17.38-5.39)/5.39)*100,
    ((44.14-18.00)/18.00)*100,
    ((58.97-27.59)/27.59)*100
]

plt.figure(figsize=(8,5))
plt.bar(metrics, improvement)

plt.ylabel("Improvement (%)")
plt.title("Improvement over Baseline")

plt.savefig(
    "improvement_graph.png",
    dpi=300
)

plt.close()

# ==========================
# GRAPH 4
# Retrieval Time
# ==========================

plt.figure(figsize=(6,5))

plt.bar(
    ["Average Query"],
    [0.0551]
)

plt.ylabel("Milliseconds")
plt.title("Retrieval Latency")

plt.savefig(
    "latency_graph.png",
    dpi=300
)

plt.close()

print("Graphs Generated!")