"""
Task 2 reporting: training curves, confusion matrix, and the GNN vs CNN
comparison table.

Run AFTER train_task2.py and train_cnn_baseline.py.

Run with:
    python plot_task2_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")


def load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_curves(data, title, outfile):
    epochs = [h["epoch"] for h in data["history"]]
    train_acc = [h["train"]["accuracy"] for h in data["history"]]
    val_acc = [h["val"]["accuracy"] for h in data["history"]]
    train_loss = [h["train"]["loss"] for h in data["history"]]
    val_loss = [h["val"]["loss"] for h in data["history"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / outfile, dpi=150)
    plt.close()
    print(f"Saved {RESULTS / outfile}")


def plot_confusion(data, outfile):
    cm = np.array(data["confusion_matrix"], dtype=float)
    genres = data["genres"]

    # Row-normalise so each row shows "of all true X, what fraction went
    # where" -- much easier to read than raw counts.
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_yticks(range(len(genres))); ax.set_yticklabels(genres)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Task 2 GNN — confusion matrix (row-normalised)")

    for i in range(len(genres)):
        for j in range(len(genres)):
            if cm_norm[i, j] > 0.01:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(RESULTS / outfile, dpi=150)
    plt.close()
    print(f"Saved {RESULTS / outfile}")


def main():
    gnn = load(RESULTS / "task2_metrics.json")
    cnn = load(RESULTS / "task2_cnn_metrics.json")

    if gnn is None:
        print("!! results/task2_metrics.json not found -- the GNN has not been")
        print("   trained yet (or it crashed). Run: python train_task2.py")
    else:
        plot_curves(gnn, "Task 2 GNN (GraphSAGE)", "task2_gnn_curves.png")
        plot_confusion(gnn, "task2_confusion_matrix.png")

    if cnn is None:
        print("!! results/task2_cnn_metrics.json not found.")
        print("   Run: python train_cnn_baseline.py")
    else:
        plot_curves(cnn, "Baseline B2 CNN (mel-spectrogram)", "task2_cnn_curves.png")

    if gnn and cnn:
        print("\n" + "=" * 62)
        print("TASK 2 COMPARISON (identical test split)")
        print("=" * 62)
        print(f"{'Model':<28}{'Accuracy':>12}{'Macro-F1':>12}")
        print("-" * 62)
        print(f"{'B1: random (10 classes)':<28}{0.10:>12.4f}{0.10:>12.4f}")
        print(f"{'B2: CNN mel-spectrogram':<28}"
              f"{cnn['test']['accuracy']:>12.4f}{cnn['test']['macro_f1']:>12.4f}")
        print(f"{'Task 2: GNN (GraphSAGE)':<28}"
              f"{gnn['test']['accuracy']:>12.4f}{gnn['test']['macro_f1']:>12.4f}")
        print("-" * 62)

        delta = gnn["test"]["macro_f1"] - cnn["test"]["macro_f1"]
        print(f"\nGNN minus CNN macro-F1: {delta:+.4f}")
        if delta > 0.02:
            print("The graph structure is adding real signal over the CNN.")
        elif delta < -0.02:
            print("The CNN is ahead. This is a legitimate, reportable finding --")
            print("discuss WHY: 30s clips give only ~19 nodes, which is a small")
            print("graph, and hand-crafted 56-dim node features discard detail")
            print("that the CNN reads straight off the spectrogram.")
        else:
            print("The two are within noise of each other. With a 199-track test")
            print("set, a gap this small is not a reliable difference -- say so")
            print("rather than claiming a winner.")

        # Per-genre comparison: where do the two models disagree?
        if gnn.get("confusion_matrix") and cnn.get("confusion_matrix"):
            import numpy as _np
            g_cm = _np.array(gnn["confusion_matrix"], dtype=float)
            c_cm = _np.array(cnn["confusion_matrix"], dtype=float)
            genres = gnn["genres"]

            g_rec = g_cm.diagonal() / (g_cm.sum(axis=1) + 1e-8)
            c_rec = c_cm.diagonal() / (c_cm.sum(axis=1) + 1e-8)

            print("\n" + "=" * 62)
            print("PER-GENRE RECALL: GNN vs CNN")
            print("=" * 62)
            print(f"{'genre':<14}{'GNN':>8}{'CNN':>8}{'diff':>9}")
            print("-" * 62)
            for i, genre in enumerate(genres):
                d = g_rec[i] - c_rec[i]
                print(f"{genre:<14}{g_rec[i]:>8.2f}{c_rec[i]:>8.2f}{d:>+9.2f}")

            gnn_wins = [genres[i] for i in range(len(genres)) if g_rec[i] - c_rec[i] > 0.1]
            cnn_wins = [genres[i] for i in range(len(genres)) if c_rec[i] - g_rec[i] > 0.1]
            print("-" * 62)
            print(f"GNN clearly better on: {gnn_wins or 'none'}")
            print(f"CNN clearly better on: {cnn_wins or 'none'}")
            if gnn_wins and cnn_wins:
                print("\nThe two models fail on DIFFERENT genres. That is the")
                print("strongest argument for fusion in Task 3: each carries signal")
                print("the other lacks. If they failed on the same tracks, combining")
                print("them would add nothing.")

            # side-by-side bar chart for the report
            fig, ax = plt.subplots(figsize=(10, 5))
            xpos = _np.arange(len(genres))
            ax.bar(xpos - 0.2, g_rec, 0.4, label="GNN (GraphSAGE)")
            ax.bar(xpos + 0.2, c_rec, 0.4, label="CNN (mel-spec)")
            ax.set_xticks(xpos); ax.set_xticklabels(genres, rotation=45, ha="right")
            ax.set_ylabel("Recall"); ax.set_ylim(0, 1)
            ax.set_title("Per-genre recall: GNN vs CNN baseline")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS / "task2_per_genre_comparison.png", dpi=150)
            plt.close()
            print(f"\nSaved {RESULTS / 'task2_per_genre_comparison.png'}")

        print("\nNote: with ~99 validation tracks, per-epoch val accuracy swings")
        print("by several points from noise alone. Judge models on the test")
        print("number from the best-val checkpoint, not on peak val accuracy.")


if __name__ == "__main__":
    main()
