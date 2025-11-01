import os
import json
import argparse
from typing import List, Dict, Any


def find_experiment_logs(logs_root: str, dataset: str, model: str) -> List[str]:
    base_dir = os.path.join(logs_root, dataset, model)
    if not os.path.isdir(base_dir):
        return []
    runs = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            runs.append(p)
    return sorted(runs)


def load_metrics_json(run_dir: str, split: str) -> Dict[str, Any]:
    # prefer metrics_<split>_<n>.json; if multiple, take the latest by mtime
    candidates = []
    for fname in os.listdir(run_dir):
        if fname.startswith(f"metrics_{split}_") and fname.endswith(".json"):
            candidates.append(os.path.join(run_dir, fname))
    if not candidates:
        return {}
    candidates.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    try:
        with open(candidates[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_params(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, "params.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def collect(args):
    runs = find_experiment_logs(args.logs_root, args.dataset, args.model)
    entries = []
    for run in runs:
        m = load_metrics_json(run, args.split)
        if not m:
            continue
        p = m.get("params", {}) or load_params(run)
        avg = m.get("average", {})
        ph = m.get("per_horizon", {})
        entry = {
            "log_dir": run,
            "n_exp": m.get("n_exp"),
            "split": args.split,
            "horizon": m.get("horizon"),
            "mae": avg.get("mae"),
            "rmse": avg.get("rmse"),
            "per_horizon_mae": ph.get("mae"),
            "per_horizon_rmse": ph.get("rmse"),
        }
        # attach key params for quick view
        for k in [
            "model_name","dataset","num_nodes","seq_len","horizon","input_dim","output_dim",
            "dropout","n_blocks","n_hidden","n_heads","spatial_flag","temporal_flag",
            "spatial_encoding","temporal_encoding","temporal_PE","GCO","CLUSTER","GCO_Thre",
            "batch_size","base_lr","lr_decay_ratio","aug","max_epochs","patience","n_exp"
        ]:
            if k in p:
                entry[k] = p[k]
        entries.append(entry)

    # sort by MAE asc
    entries.sort(key=lambda e: (e.get("mae") is None, e.get("mae")))

    # write txt
    txt_path = os.path.join(args.output_dir, f"summary_{args.split}.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries):
            f.write(
                f"[{i+1}] mae={e.get('mae'):.4f} rmse={e.get('rmse'):.4f} "
                f"dir={e.get('log_dir')} n_exp={e.get('n_exp')}\n"
            )

    # optional CSV/JSON
    if args.csv:
        import csv
        csv_path = os.path.join(args.output_dir, f"summary_{args.split}.csv")
        keys = [
            "log_dir","n_exp","split","horizon","mae","rmse",
            "model_name","dataset","num_nodes","seq_len","input_dim","output_dim",
            "dropout","n_blocks","n_hidden","n_heads","spatial_flag","temporal_flag",
            "spatial_encoding","temporal_encoding","temporal_PE","GCO","CLUSTER","GCO_Thre",
            "batch_size","base_lr","lr_decay_ratio","aug","max_epochs","patience"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for e in entries:
                writer.writerow({k: e.get(k) for k in keys})

    if args.json:
        json_path = os.path.join(args.output_dir, f"summary_{args.split}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_root", type=str, default="./logs")
    parser.add_argument("--dataset", type=str, default="SINPA")
    parser.add_argument("--model", type=str, default="DeepPA")
    parser.add_argument("--split", type=str, default="test", choices=["train","val","test"]) 
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()