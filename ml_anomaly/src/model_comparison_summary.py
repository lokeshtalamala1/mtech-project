import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Step 1: script started")

# =========================
# 1. Output folder
# =========================
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)
print("Step 2: output folder ready ->", outdir)

# =========================
# 2. Final summary table
# =========================
summary = pd.DataFrame([
    ["Isolation Forest", "Traditional ML", "Row-level", 0.5085, 1.0000, 0.6742, 0.9964, 0.8525],
    ["One-Class SVM", "Traditional ML", "Row-level", 0.4839, 1.0000, 0.6522, 0.9990, 0.9447],
    ["LOF", "Traditional ML", "Row-level", 0.0000, 0.0000, 0.0000, 0.9159, 0.1084],
    ["Autoencoder", "Deep Learning", "Row-level", 0.5085, 1.0000, 0.6742, 0.9975, 0.8925],
    ["LSTM Autoencoder", "Deep Learning / UEBA", "Sequence-level", 0.7611, 0.7049, 0.7319, 0.7948, 0.7393],
    ["GraphSAGE", "Graph-based ML", "Node-level", 0.0000, 0.0000, 0.0000, 0.6471, 0.1964],
], columns=[
    "Model", "Model_Type", "Evaluation_Level",
    "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"
])

summary_csv = os.path.join(outdir, "model_comparison_summary.csv")
summary.to_csv(summary_csv, index=False)
print("Step 3: summary csv saved ->", summary_csv)

# =========================
# 3. Overall comparison plot
# =========================
plt.figure(figsize=(10, 6))

x = np.arange(len(summary))

plt.plot(x, summary["F1_Score"], marker="o", label="F1-score")
plt.plot(x, summary["ROC_AUC"], marker="o", label="ROC-AUC")
plt.plot(x, summary["PR_AUC"], marker="o", label="PR-AUC")

plt.xticks(x, summary["Model"], rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Overall Model Comparison Summary")
plt.legend()
plt.tight_layout()

overall_plot = os.path.join(outdir, "overall_model_comparison.png")
plt.savefig(overall_plot, dpi=200)
plt.close()

print("Step 4: overall plot saved ->", overall_plot)

# =========================
# 4. Row-level models plot
# =========================
row_models = summary[summary["Evaluation_Level"] == "Row-level"]

plt.figure(figsize=(8, 6))

x_row = np.arange(len(row_models))

plt.plot(x_row, row_models["F1_Score"], marker="o", label="F1-score")
plt.plot(x_row, row_models["ROC_AUC"], marker="o", label="ROC-AUC")
plt.plot(x_row, row_models["PR_AUC"], marker="o", label="PR-AUC")

plt.xticks(x_row, row_models["Model"], rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Row-Level Model Comparison")
plt.legend()
plt.tight_layout()

row_plot = os.path.join(outdir, "row_level_model_comparison.png")
plt.savefig(row_plot, dpi=200)
plt.close()

print("Step 5: row-level plot saved ->", row_plot)

print("\nDone.")
print("Generated files:")
print("-", summary_csv)
print("-", overall_plot)
print("-", row_plot)