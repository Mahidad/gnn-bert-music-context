"""
Reads results/task1_metrics.json (written by train_task1.py) and plots
Macro-F1 / Micro-F1 vs. epoch -- this is the curve the assignment brief
asks for as a Task 1 deliverable. Run this after train_task1.py finishes.
"""

import json
import matplotlib.pyplot as plt


def main():
    with open("results/task1_metrics.json") as f:
        data = json.load(f)

    history = data["history"]
    epochs = [h["epoch"] for h in history]
    macro_f1 = [h["macro_f1"] for h in history]
    micro_f1 = [h["micro_f1"] for h in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, macro_f1, marker="o", label="Macro-F1")
    plt.plot(epochs, micro_f1, marker="o", label="Micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Task 1: BERT Tag Classifier -- F1 vs. Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task1_f1_curve.png", dpi=150)
    print("Saved plot to results/task1_f1_curve.png")


if __name__ == "__main__":
    main()
