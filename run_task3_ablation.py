"""
Run the full Task 3 ablation grid and print the comparison table.

Trains every fusion variant under both caption conditions, then assembles
the table the assignment asks for. Skips any combination whose result file
already exists, so it is safe to interrupt and re-run.

Run with:
    python run_task3_ablation.py
    python run_task3_ablation.py --masked-only
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESULTS = Path("results/task3")

# gnn_only ignores captions entirely, so running it twice would train the
# identical model on identical data. It is listed once, under "masked",
# and reused as the reference row for both conditions.
GRID = [
    ("gnn_only", "masked"),
    ("bert_only", "masked"),
    ("bert_only", "unmasked"),
    ("concat", "masked"),
    ("concat", "unmasked"),
    ("cross_attention", "masked"),
    ("cross_attention", "unmasked"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="retrain even if a result file exists")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    # Flags passed straight through to train_task3.py, so a whole ablation
    # grid can be run under lighter settings without editing anything.
    passthrough = []
    if args.batch_size:  passthrough += ["--batch-size", str(args.batch_size)]
    if args.max_length:  passthrough += ["--max-length", str(args.max_length)]
    if args.freeze_bert: passthrough += ["--freeze-bert"]
    if args.amp:         passthrough += ["--amp"]

    grid = [(v, m) for v, m in GRID if not (args.masked_only and m == "unmasked")]

    for variant, mode in grid:
        out = RESULTS / f"{variant}_{mode}.json"
        if out.exists() and not args.force:
            print(f"[skip] {variant} / {mode} already done")
            continue
        print(f"\n{'=' * 62}\nTRAINING {variant} / {mode}\n{'=' * 62}")
        subprocess.run([sys.executable, "train_task3.py",
                        "--variant", variant, "--caption-mode", mode]
                       + passthrough, check=False)

    # ---------------- assemble the table ----------------
    rows = []
    for variant, mode in grid:
        out = RESULTS / f"{variant}_{mode}.json"
        if out.exists():
            with open(out) as f:
                d = json.load(f)
            rows.append((variant, mode, d["test"], d.get("test_tuned", d["test"])))

    if not rows:
        print("\nNo results to tabulate.")
        return

    print("\n" + "=" * 86)
    print("TASK 3 ABLATION (identical test split)")
    print("=" * 86)
    print(f"{'variant':<18}{'captions':<12}{'MacroF1':>10}{'MacroF1*':>10}"
          f"{'MicroF1*':>10}{'AUC-PR':>10}")
    print(f"{'':<30}{'@0.5':>10}{'tuned':>10}{'tuned':>10}{'':>10}")
    print("-" * 86)
    for variant, mode, m, mt in rows:
        label = "n/a (audio)" if variant == "gnn_only" else mode
        print(f"{variant:<18}{label:<12}{m['macro_f1']:>10.4f}{mt['macro_f1']:>10.4f}"
              f"{mt['micro_f1']:>10.4f}{m['auc_pr']:>10.4f}")
    print("-" * 86)
    print("* per-tag thresholds fitted on validation, applied unchanged to test.")
    print("  AUC-PR is threshold-free, so it is unaffected by tuning.")

    def get(variant, mode, tuned=True):
        for v, m, met, met_tuned in rows:
            if v == variant and m == mode:
                return met_tuned if tuned else met
        return None

    for mode in (["masked"] if args.masked_only else ["masked", "unmasked"]):
        b, x = get("bert_only", mode), get("cross_attention", mode)
        if b and x:
            d = x["macro_f1"] - b["macro_f1"]
            da = x["auc_pr"] - b["auc_pr"]
            print(f"\n[{mode}] cross-attention minus BERT-only: "
                  f"macro-F1 {d:+.4f}, AUC-PR {da:+.4f}")
            if d > 0.02:
                print("  Adding audio structure helps: the graph carries information")
                print("  the caption does not.")
            elif d < -0.02:
                print("  Fusion HURTS here. Usually this means the text alone already")
                print("  answers the task, and the extra parameters just add noise.")
            else:
                print("  Within noise -- no reliable difference.")

        c, x2 = get("concat", mode), get("cross_attention", mode)
        if c and x2:
            d = x2["macro_f1"] - c["macro_f1"]
            print(f"[{mode}] cross-attention minus early-concat macro-F1: {d:+.4f}")
            if abs(d) <= 0.02:
                print("  Interaction is not beating simple concatenation. Worth saying")
                print("  plainly -- a negative architectural result is still a result.")

    bm, bu = get("bert_only", "masked"), get("bert_only", "unmasked")
    if bm and bu:
        drop = bu["macro_f1"] - bm["macro_f1"]
        print(f"\nBERT-only drop from unmasked to masked: {drop:+.4f}")
        print("  This is the size of the extraction shortcut. A large drop confirms")
        print("  the unmasked task was substantially phrase-matching rather than")
        print("  music understanding.")


if __name__ == "__main__":
    main()
