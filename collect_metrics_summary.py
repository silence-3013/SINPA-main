import argparse
import os
import glob
import re
from datetime import datetime


def parse_metrics_txt(path):
    split = None
    horizon = None
    avg_mae = None
    avg_rmse = None
    per_horizon = []
    windows = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                m = re.match(r"^Split:\s*(\w+),\s*Horizon:\s*(\d+)", line)
                if m:
                    split = m.group(1)
                    horizon = int(m.group(2))
                    continue
                m = re.match(r"^Average MAE:\s*([0-9.]+),\s*Average RMSE:\s*([0-9.]+)", line)
                if m:
                    avg_mae = float(m.group(1))
                    avg_rmse = float(m.group(2))
                    continue
                m = re.match(r"^h=(\d+)\s+MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+)$", line)
                if m:
                    per_horizon.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                    continue
                m = re.match(r"^0-3 MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+)$", line)
                if m:
                    windows["0-3"] = (float(m.group(1)), float(m.group(2)))
                    continue
                m = re.match(r"^4-7 MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+)$", line)
                if m:
                    windows["4-7"] = (float(m.group(1)), float(m.group(2)))
                    continue
                m = re.match(r"^8-11 MAE:\s*([0-9.]+),\s*RMSE:\s*([0-9.]+)$", line)
                if m:
                    windows["8-11"] = (float(m.group(1)), float(m.group(2)))
                    continue
    except Exception:
        pass
    return {
        "split": split,
        "horizon": horizon,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "per_horizon": per_horizon,
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect metrics_<split>_*.txt under logs and write summaries at repo root.")
    parser.add_argument("--logs_root", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Which split to collect")
    parser.add_argument("--output", type=str, default=None, help="Output summary path (default: ./summary_<split>.txt)")
    parser.add_argument("--csv", type=str, default="True", choices=["True", "False"], help="Also write summary_<split>.csv with params columns")
    parser.add_argument("--json", type=str, default="False", choices=["True", "False"], help="Also write summary_<split>.json containing entries list")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if args.output is None:
        out_path = os.path.join(repo_root, f"summary_{args.split}.txt")
    else:
        out_path = os.path.abspath(args.output)

    logs_root = os.path.abspath(args.logs_root)
    pattern = os.path.join(logs_root, "**", f"metrics_{args.split}_*.txt")
    files = glob.glob(pattern, recursive=True)

    entries = []
    for fpath in files:
        rel = os.path.relpath(fpath, logs_root)
        parts = rel.split(os.sep)
        dataset = parts[0] if len(parts) > 0 else "?"
        model = parts[1] if len(parts) > 1 else "?"
        folder = parts[2] if len(parts) > 2 else "?"
        m = re.search(r"metrics_{}_([0-9]+)\.txt".format(re.escape(args.split)), os.path.basename(fpath))
        n_exp = int(m.group(1)) if m else None

        parsed = parse_metrics_txt(fpath)
        if parsed["avg_mae"] is None or parsed["avg_rmse"] is None:
            continue
        # try to read params.json in the same directory
        params = {}
        try:
            pjson = os.path.join(os.path.dirname(fpath), "params.json")
            if os.path.exists(pjson):
                import json
                with open(pjson, "r", encoding="utf-8") as pf:
                    params = json.load(pf)
        except Exception:
            params = {}
        entries.append({
            "dataset": dataset,
            "model": model,
            "folder": folder,
            "n_exp": n_exp,
            "avg_mae": parsed["avg_mae"],
            "avg_rmse": parsed["avg_rmse"],
            "horizon": parsed["horizon"],
            "file": rel,
            "params": params,
        })

    # sort by avg_mae then avg_rmse
    entries.sort(key=lambda x: (x["avg_mae"], x["avg_rmse"]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"Summary for split='{args.split}'\n")
        out.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Logs root: {logs_root}\n")
        out.write(f"Found files: {len(files)}, Parsed valid entries: {len(entries)}\n")
        out.write("\nBest (sorted by MAE then RMSE):\n")
        if not entries:
            out.write("No metrics files found. Run test to generate metrics_<split>_*.txt.\n")
        else:
            out.write("dataset/model/folder | exp | horizon | avg_mae | avg_rmse | impl/adapt/alpha/tau/levels | file\n")
            for e in entries:
                p = e.get("params", {})
                impl = str(p.get("gco_impl", "?"))
                adapt = str(p.get("gco_adaptive", "?"))
                alpha = str(p.get("gco_alpha", "?"))
                tau = str(p.get("gco_tau", "?"))
                levels = str(p.get("gco_wavelet_levels", "?"))
                param_str = f"{impl}/{adapt}/{alpha}/{tau}/{levels}"
                out.write(f"{e['dataset']}/{e['model']}/{e['folder']} | {e['n_exp']} | {e['horizon']} | {e['avg_mae']:.4f} | {e['avg_rmse']:.4f} | {param_str} | {e['file']}\n")

    print(f"Summary written to: {out_path}")

    # optional CSV output with params columns
    if args.csv == "True":
        csv_path = os.path.join(repo_root, f"summary_{args.split}.csv")
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow([
                "dataset","model","folder","exp","horizon","avg_mae","avg_rmse",
                "gco_impl","gco_adaptive","gco_alpha","gco_tau","gco_wavelet_levels",
                "seed","batch_size","base_lr","max_epochs","patience","file"
            ])
            for e in entries:
                p = e.get("params", {})
                writer.writerow([
                    e.get("dataset"), e.get("model"), e.get("folder"), e.get("n_exp"), e.get("horizon"),
                    f"{e.get('avg_mae'):.4f}", f"{e.get('avg_rmse'):.4f}",
                    p.get("gco_impl"), p.get("gco_adaptive"), p.get("gco_alpha"), p.get("gco_tau"), p.get("gco_wavelet_levels"),
                    p.get("seed"), p.get("batch_size"), p.get("base_lr"), p.get("max_epochs"), p.get("patience"),
                    e.get("file")
                ])
        print(f"CSV summary written to: {csv_path}")

    # optional JSON output
    if args.json == "True":
        json_path = os.path.join(repo_root, f"summary_{args.split}.json")
        import json as _json
        with open(json_path, "w", encoding="utf-8") as jf:
            _json.dump({
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "split": args.split,
                "logs_root": logs_root,
                "entries": entries,
            }, jf, ensure_ascii=False, indent=2)
        print(f"JSON summary written to: {json_path}")


if __name__ == "__main__":
    main()