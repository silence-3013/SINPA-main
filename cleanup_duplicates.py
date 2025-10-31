import argparse
import hashlib
import os
import re
import shutil
import time


def hash_file(path: str, algo: str = "sha256", chunk_size: int = 8192) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_dup_filename(fname: str) -> bool:
    base, _ = os.path.splitext(fname)
    return bool(re.match(r"^.* \(\d+\)$", base))


def canonical_name(fname: str) -> str:
    base, ext = os.path.splitext(fname)
    m = re.match(r"^(.*) \(\d+\)$", base)
    if not m:
        return fname
    return f"{m.group(1)}{ext}"


def should_skip_dir(dirname: str, excludes: set) -> bool:
    return dirname in excludes


def main():
    parser = argparse.ArgumentParser(description="Safely clean duplicate files like 'name (2).ext'.")
    parser.add_argument("--root", type=str, default=os.path.dirname(os.path.abspath(__file__)), help="Repo root to scan")
    parser.add_argument("--apply", action="store_true", help="Apply deletions for identical duplicates")
    parser.add_argument("--backup", action="store_true", help="Backup identical duplicates before deletion")
    parser.add_argument("--backup_dir", type=str, default=None, help="Backup directory (default: ./backup/duplicates/<timestamp>/)")
    parser.add_argument("--exclude", type=str, default="__pycache__,.git,logs,venv,.idea", help="Comma-separated directories to exclude")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    excludes = set([x.strip() for x in args.exclude.split(",") if x.strip()])

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = args.backup_dir or os.path.join(root, "backup", "duplicates", ts)

    total = 0
    identical = 0
    deleted = 0
    diff = 0
    missing_canonical = 0

    print(f"Scanning: {root}")
    print(f"Exclude dirs: {sorted(excludes)}")
    actions = []

    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d, excludes)]

        for fname in filenames:
            if not is_dup_filename(fname):
                continue

            total += 1
            dup_path = os.path.join(dirpath, fname)
            canon_fname = canonical_name(fname)
            canon_path = os.path.join(dirpath, canon_fname)

            if not os.path.exists(canon_path):
                missing_canonical += 1
                actions.append((dup_path, canon_path, "MISSING_CANON"))
                continue

            # quick check: size
            try:
                dup_size = os.path.getsize(dup_path)
                canon_size = os.path.getsize(canon_path)
            except OSError:
                actions.append((dup_path, canon_path, "STAT_ERROR"))
                continue

            if dup_size != canon_size:
                diff += 1
                actions.append((dup_path, canon_path, "DIFF_SIZE"))
                continue

            # strong check: hash
            try:
                dup_hash = hash_file(dup_path)
                canon_hash = hash_file(canon_path)
            except OSError:
                actions.append((dup_path, canon_path, "READ_ERROR"))
                continue

            if dup_hash == canon_hash:
                identical += 1
                if args.apply:
                    # optional backup
                    if args.backup:
                        rel = os.path.relpath(dup_path, root)
                        backup_path = os.path.join(backup_root, rel)
                        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                        shutil.copy2(dup_path, backup_path)
                        print(f"Backed up: {dup_path} -> {backup_path}")
                    try:
                        os.remove(dup_path)
                        deleted += 1
                        actions.append((dup_path, canon_path, "DELETED"))
                    except OSError:
                        actions.append((dup_path, canon_path, "DELETE_ERROR"))
                else:
                    actions.append((dup_path, canon_path, "IDENTICAL"))
            else:
                diff += 1
                actions.append((dup_path, canon_path, "DIFF_HASH"))

    # report
    print("\n=== Duplicate Scan Summary ===")
    print(f"Found duplicates: {total}")
    print(f"Identical to canonical: {identical}")
    print(f"Different from canonical: {diff}")
    print(f"Canonical missing: {missing_canonical}")
    if args.apply:
        print(f"Deleted identical duplicates: {deleted}")
        if args.backup:
            print(f"Backups located at: {backup_root}")
    else:
        print("Dry-run (no deletions). Use --apply to delete identical duplicates.")

    # detailed actions
    print("\nDetailed actions:")
    for dup_path, canon_path, status in actions:
        print(f"[{status}] {dup_path} -> {canon_path}")


if __name__ == "__main__":
    main()