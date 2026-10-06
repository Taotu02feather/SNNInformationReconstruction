import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "experiments"))

OUT_CURVE_DIR = os.path.join(SCRIPT_DIR, "bptt_tess_curves")
OUT_SUMMARY_DIR = os.path.join(SCRIPT_DIR, "bptt_tess_summary")

os.makedirs(OUT_CURVE_DIR, exist_ok=True)
os.makedirs(OUT_SUMMARY_DIR, exist_ok=True)



VALID_FOLDERS = {
    "CIFAR10_VGG_BPTT_",
    "CIFAR10_VGG_TESS_",
    "CIFAR100_VGG_BPTT_",
    "CIFAR100_VGG_TESS_",
    "CIFAR10DVS_VGG_BPTT_",
    "CIFAR10DVS_VGG_TESS_",
    "DVSGesture_BPTT_",
    "DVSGesture_TESS_",
}



def parse_compare_folder_name(folder_name):
    """
    只接受白名单中的 8 个目录名。
    例如：
      CIFAR10_VGG_TESS_   Correct
      CIFAR10_VGG_TESS    Wrong
    """
    if folder_name not in VALID_FOLDERS:
        return None

    if folder_name.endswith("_VGG_BPTT_"):
        dataset = folder_name[:-len("_VGG_BPTT_")]
        method = "BPTT"
    elif folder_name.endswith("_VGG_TESS_"):
        dataset = folder_name[:-len("_VGG_TESS_")]
        method = "TESS"
    elif folder_name.endswith("_BPTT_"):
        dataset = folder_name[:-len("_BPTT_")]
        method = "BPTT"
    elif folder_name.endswith("_TESS_"):
        dataset = folder_name[:-len("_TESS_")]
        method = "TESS"
    else:
        return None

    return dataset, method


def parse_log_file(log_path):
    train_re = re.compile(r"@Training\s+\*\s+Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)\s+Loss\s+([0-9.]+)")
    test_re = re.compile(r"@Testing\s+\*\s+Acc@1\s+([0-9.]+)\s+Acc@5\s+([0-9.]+)\s+Loss\s+([0-9.]+)")
    best_re = re.compile(r"Best acc at epoch\s+(\d+):\s+([0-9.eE+-]+)")
    lr_re = re.compile(r"Last learning rate:\s+\[([0-9.eE+-]+)\]")
    epoch_re = re.compile(r"Epoch\s+(\d+)")

    current_epoch = None
    epoch_dict = {}
    best_epoch = None
    best_acc = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Epoch" in line and "..." in line:
                m = epoch_re.search(line)
                if m:
                    current_epoch = int(m.group(1))
                    if current_epoch not in epoch_dict:
                        epoch_dict[current_epoch] = {
                            "epoch": current_epoch,
                            "train_acc1": None,
                            "train_acc5": None,
                            "train_loss": None,
                            "test_acc1": None,
                            "test_acc5": None,
                            "test_loss": None,
                            "lr": None,
                        }
                continue

            m = train_re.search(line)
            if m and current_epoch is not None:
                epoch_dict[current_epoch]["train_acc1"] = float(m.group(1))
                epoch_dict[current_epoch]["train_acc5"] = float(m.group(2))
                epoch_dict[current_epoch]["train_loss"] = float(m.group(3))
                continue

            m = test_re.search(line)
            if m and current_epoch is not None:
                epoch_dict[current_epoch]["test_acc1"] = float(m.group(1))
                epoch_dict[current_epoch]["test_acc5"] = float(m.group(2))
                epoch_dict[current_epoch]["test_loss"] = float(m.group(3))
                continue

            m = best_re.search(line)
            if m:
                best_epoch = int(m.group(1))
                best_acc = float(m.group(2))
                continue

            m = lr_re.search(line)
            if m and current_epoch is not None:
                epoch_dict[current_epoch]["lr"] = float(m.group(1))
                continue

    df = pd.DataFrame([epoch_dict[k] for k in sorted(epoch_dict.keys())])

    if df.empty:
        summary = {
            "best_acc": None,
            "best_epoch": None,
            "final_epoch": None,
            "final_train_acc1": None,
            "final_test_acc1": None,
            "final_train_loss": None,
            "final_test_loss": None,
            "final_lr": None,
        }
        return df, summary

    final_row = df.iloc[-1]
    summary = {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_epoch": int(final_row["epoch"]),
        "final_train_acc1": final_row["train_acc1"],
        "final_test_acc1": final_row["test_acc1"],
        "final_train_loss": final_row["train_loss"],
        "final_test_loss": final_row["test_loss"],
        "final_lr": final_row["lr"],
    }
    return df, summary


def main():
    summary_rows = []

    for folder in sorted(os.listdir(EXP_ROOT)):
        folder_path = os.path.join(EXP_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_compare_folder_name(folder)
        if parsed is None:
            print(f"[SKIP][compare] invalid folder name: {folder}")
            continue

        dataset, method = parsed
        log_path = os.path.join(folder_path, "log.log")
        if not os.path.exists(log_path):
            print(f"[SKIP] log not found: {log_path}")
            continue

        curve_df, summary = parse_log_file(log_path)
        curve_df.insert(0, "method", method)
        curve_df.insert(0, "dataset", dataset)

        curve_name = f"{dataset}_{method}_curve.csv"
        curve_df.to_csv(os.path.join(OUT_CURVE_DIR, curve_name), index=False)

        summary["dataset"] = dataset
        summary["method"] = method
        summary["folder"] = folder
        summary_rows.append(summary)

        print(f"[OK] {folder} -> {curve_name}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "method"])
    summary_path = os.path.join(OUT_SUMMARY_DIR, "bptt_tess_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    print("EXP_ROOT =", EXP_ROOT)
    main()