import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 输入：你前面已经生成好的 summary =====
ALPHA_SUMMARY = os.path.join(SCRIPT_DIR, "alpha_post_summary", "alpha_ablation_summary.csv")
COMPARE_SUMMARY = os.path.join(SCRIPT_DIR, "bptt_tess_summary", "bptt_tess_summary.csv")

# ===== 输出目录 =====
OUT_DIR = os.path.join(SCRIPT_DIR, "paper_compare")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# 论文原始结果（来自 main.pdf 的 Table II / Table III）
# Table II: alpha_post ablation
# Table III: BPTT / S-TLLR / TESS baseline
# =========================================================
PAPER_ALPHA = [
    {"dataset": "CIFAR10DVS", "alpha_post": -1, "paper_acc": 75.00, "paper_std": 0.69},
    {"dataset": "CIFAR10DVS", "alpha_post":  0, "paper_acc": 75.00, "paper_std": 0.65},
    {"dataset": "CIFAR10DVS", "alpha_post":  1, "paper_acc": 74.36, "paper_std": 0.87},

    {"dataset": "DVSGesture", "alpha_post": -1, "paper_acc": 98.56, "paper_std": 0.41},
    {"dataset": "DVSGesture", "alpha_post":  0, "paper_acc": 98.33, "paper_std": 0.57},
    {"dataset": "DVSGesture", "alpha_post":  1, "paper_acc": 98.56, "paper_std": 0.31},

    {"dataset": "CIFAR10", "alpha_post": -1, "paper_acc": 89.93, "paper_std": 0.31},
    {"dataset": "CIFAR10", "alpha_post":  0, "paper_acc": 91.99, "paper_std": 0.19},
    {"dataset": "CIFAR10", "alpha_post":  1, "paper_acc": 92.55, "paper_std": 0.16},

    {"dataset": "CIFAR100", "alpha_post": -1, "paper_acc": 62.49, "paper_std": 1.05},
    {"dataset": "CIFAR100", "alpha_post":  0, "paper_acc": 68.19, "paper_std": 0.55},
    {"dataset": "CIFAR100", "alpha_post":  1, "paper_acc": 70.00, "paper_std": 0.34},
]

PAPER_COMPARE = [
    {"dataset": "CIFAR10DVS", "method": "BPTT",   "paper_acc": 76.40, "paper_std": 0.66},
    {"dataset": "CIFAR10DVS", "method": "S-TLLR", "paper_acc": 75.14, "paper_std": 1.37},
    {"dataset": "CIFAR10DVS", "method": "TESS",   "paper_acc": 75.00, "paper_std": 0.65},

    {"dataset": "DVSGesture", "method": "BPTT",   "paper_acc": 97.95, "paper_std": 0.68},
    {"dataset": "DVSGesture", "method": "S-TLLR", "paper_acc": 98.48, "paper_std": 0.37},
    {"dataset": "DVSGesture", "method": "TESS",   "paper_acc": 98.56, "paper_std": 0.31},

    {"dataset": "CIFAR10", "method": "BPTT",   "paper_acc": 92.55, "paper_std": 0.06},
    {"dataset": "CIFAR10", "method": "S-TLLR", "paper_acc": 91.88, "paper_std": 0.28},
    {"dataset": "CIFAR10", "method": "TESS",   "paper_acc": 92.55, "paper_std": 0.16},

    {"dataset": "CIFAR100", "method": "BPTT",   "paper_acc": 69.28, "paper_std": 0.37},
    {"dataset": "CIFAR100", "method": "S-TLLR", "paper_acc": 68.00, "paper_std": 0.71},
    {"dataset": "CIFAR100", "method": "TESS",   "paper_acc": 70.00, "paper_std": 0.34},
]


def normalize_dataset_name(x: str) -> str:
    x = str(x).strip()
    mapping = {
        "DVS Gesture": "DVSGesture",
        "IBM DVS Gesture": "DVSGesture",
        "CIFAR10-DVS": "CIFAR10DVS",
        "CIFAR10DVS": "CIFAR10DVS",
        "CIFAR10": "CIFAR10",
        "CIFAR100": "CIFAR100",
        "DVSGesture": "DVSGesture",
    }
    return mapping.get(x, x)


def write_markdown_table(df: pd.DataFrame, path: str, title: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def build_alpha_compare():
    if not os.path.exists(ALPHA_SUMMARY):
        raise FileNotFoundError(f"Not found: {ALPHA_SUMMARY}")

    my_df = pd.read_csv(ALPHA_SUMMARY)
    my_df["dataset"] = my_df["dataset"].map(normalize_dataset_name)

    # 你可以在这里改成 final_test_acc1
    metric_col = "final_test_acc1"
    if metric_col not in my_df.columns:
        raise ValueError(f"{ALPHA_SUMMARY} 中找不到列: {metric_col}")

    my_df = my_df[["dataset", "alpha_post", metric_col]].copy()
    my_df = my_df.rename(columns={metric_col: "my_acc"})

    paper_df = pd.DataFrame(PAPER_ALPHA)

    merged = pd.merge(
        paper_df,
        my_df,
        on=["dataset", "alpha_post"],
        how="left"
    )

    merged["diff_my_minus_paper"] = merged["my_acc"] - merged["paper_acc"]

    out_csv = os.path.join(OUT_DIR, "alpha_post_vs_paper.csv")
    merged.to_csv(out_csv, index=False)

    md_df = merged.copy()
    md_df["paper_result"] = md_df.apply(
        lambda r: f"{r['paper_acc']:.2f} ± {r['paper_std']:.2f}", axis=1
    )
    md_df["my_result"] = md_df["my_acc"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    md_df["delta"] = md_df["diff_my_minus_paper"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    md_df = md_df[["dataset", "alpha_post", "paper_result", "my_result", "delta"]]

    out_md = os.path.join(OUT_DIR, "alpha_post_vs_paper.md")
    write_markdown_table(md_df, out_md, "Alpha ablation: paper vs my results")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


def build_bptt_tess_compare():
    if not os.path.exists(COMPARE_SUMMARY):
        raise FileNotFoundError(f"Not found: {COMPARE_SUMMARY}")

    my_df = pd.read_csv(COMPARE_SUMMARY)
    my_df["dataset"] = my_df["dataset"].map(normalize_dataset_name)

    # 若旧 summary 里没有 S-TLLR，这里只会匹配到 BPTT / TESS
    # 如果后续加了 S-TLLR summary，它会自动并入
    metric_col = "best_acc"
    if metric_col not in my_df.columns:
        raise ValueError(f"{COMPARE_SUMMARY} 中找不到列: {metric_col}")

    my_df = my_df[["dataset", "method", metric_col]].copy()
    my_df = my_df.rename(columns={metric_col: "my_acc"})

    paper_df = pd.DataFrame(PAPER_COMPARE)

    merged = pd.merge(
        paper_df,
        my_df,
        on=["dataset", "method"],
        how="left"
    )

    merged["diff_my_minus_paper"] = merged["my_acc"] - merged["paper_acc"]

    out_csv = os.path.join(OUT_DIR, "bptt_tess_vs_paper.csv")
    merged.to_csv(out_csv, index=False)

    md_df = merged.copy()
    md_df["paper_result"] = md_df.apply(
        lambda r: f"{r['paper_acc']:.2f} ± {r['paper_std']:.2f}", axis=1
    )
    md_df["my_result"] = md_df["my_acc"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    md_df["delta"] = md_df["diff_my_minus_paper"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
    md_df = md_df[["dataset", "method", "paper_result", "my_result", "delta"]]

    out_md = os.path.join(OUT_DIR, "bptt_tess_vs_paper.md")
    write_markdown_table(md_df, out_md, "BPTT / S-TLLR / TESS: paper vs my results")

    print(f"[OK] {out_csv}")
    print(f"[OK] {out_md}")


if __name__ == "__main__":
    build_alpha_compare()
    build_bptt_tess_compare()
    print("[DONE] All comparison files saved in:", OUT_DIR)