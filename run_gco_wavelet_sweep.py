import argparse
import os
import sys
import subprocess
import shlex
import json
import time
from datetime import datetime


def parse_list(arg_str, cast=float):
    vals = []
    for part in arg_str.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(cast(part))
        except Exception:
            pass
    return vals


def build_command(python_exe,
                  dataset,
                  gpu,
                  n_exp,
                  seed,
                  alpha,
                  tau,
                  levels,
                  batch_size,
                  max_epochs,
                  patience,
                  wandb,
                  wandb_mode):
    cmd = [
        python_exe,
        os.path.join('experiments', 'DeepPA', 'main.py'),
        '--dataset', dataset,
        '--mode', 'train',
        '--gpu', str(gpu),
        '--GCO', 'True',
        '--gco_impl', 'wavelet',
        '--gco_adaptive', 'True',
        '--gco_alpha', str(alpha),
        '--gco_tau', str(tau),
        '--gco_wavelet_levels', str(levels),
        '--batch_size', str(batch_size),
        '--max_epochs', str(max_epochs),
        '--patience', str(patience),
        '--n_exp', str(n_exp),
        '--seed', str(seed),
        '--wandb', 'True' if wandb else 'False',
        '--wandb_mode', wandb_mode,
    ]
    return cmd


def detect_log_dir(stdout_text):
    # main.py prints args.log_dir explicitly, typically like: ./logs/<dataset>/<model>/<folder>
    lines = stdout_text.splitlines()
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('./logs/') or line_stripped.startswith('.\\logs\\') or line_stripped.startswith('.\\logs/'):
            return line_stripped
    # also try to find a line that looks like a path with 'logs'
    for line in lines:
        if 'logs' in line:
            return line.strip()
    return None


def normalize_path(p):
    # Convert printed path to OS-specific absolute path
    p = p.replace('/', os.sep).replace('\\', os.sep)
    if p.startswith('.' + os.sep):
        p = p[2:]
    abs_p = os.path.abspath(p)
    return abs_p


def write_params_json(log_dir, params):
    try:
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, 'params.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        return out_path
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Run a sweep of wavelet+adaptive GCO parameters and record results.')
    parser.add_argument('--dataset', type=str, default='base')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--alphas', type=str, default='5,10,20')
    parser.add_argument('--taus', type=str, default='-0.5,0.0,0.5')
    parser.add_argument('--levels', type=str, default='1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n_exp_start', type=int, default=1)
    parser.add_argument('--wandb', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['offline', 'online', 'disabled'])
    parser.add_argument('--dry_run', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('--summary_after', type=str, default='True', choices=['True', 'False'])
    args = parser.parse_args()

    seeds = parse_list(args.seeds, int)
    alphas = parse_list(args.alphas, float)
    taus = parse_list(args.taus, float)
    levels_list = parse_list(args.levels, int)

    python_exe = sys.executable if sys.executable else 'python'
    wandb_flag = (args.wandb == 'True')
    dry_run_flag = (args.dry_run == 'True')
    summary_flag = (args.summary_after == 'True')

    runs_csv = os.path.abspath('sweep_runs.csv')
    with open(runs_csv, 'w', encoding='utf-8') as f:
        f.write('timestamp,dataset,n_exp,seed,alpha,tau,levels,batch_size,max_epochs,patience,wandb,wandb_mode,cmd,log_dir,params_json,metrics_txt,status\n')

    n_exp = args.n_exp_start
    total = len(seeds) * len(alphas) * len(taus) * len(levels_list)
    idx = 0
    for seed in seeds:
        for alpha in alphas:
            for tau in taus:
                for lvl in levels_list:
                    idx += 1
                    cmd_list = build_command(python_exe, args.dataset, args.gpu, n_exp, seed, alpha, tau, lvl,
                                             args.batch_size, args.max_epochs, args.patience, wandb_flag, args.wandb_mode)
                    cmd_str = ' '.join([shlex.quote(c) for c in cmd_list])
                    print(f"[{idx}/{total}] n_exp={n_exp} seed={seed} alpha={alpha} tau={tau} levels={lvl}")
                    print(cmd_str)

                    log_dir_printed = None
                    params_json_path = None
                    metrics_txt_path = None
                    status = 'pending'
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    if dry_run_flag:
                        status = 'dry_run'
                    else:
                        try:
                            proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.path.abspath('.'))
                            out = proc.stdout or ''
                            log_dir_printed = detect_log_dir(out)
                            if log_dir_printed:
                                log_dir = normalize_path(log_dir_printed)
                                params = {
                                    'dataset': args.dataset,
                                    'n_exp': n_exp,
                                    'seed': seed,
                                    'gco_impl': 'wavelet',
                                    'gco_adaptive': True,
                                    'gco_alpha': alpha,
                                    'gco_tau': tau,
                                    'gco_wavelet_levels': lvl,
                                    'batch_size': args.batch_size,
                                    'max_epochs': args.max_epochs,
                                    'patience': args.patience,
                                    'wandb': wandb_flag,
                                    'wandb_mode': args.wandb_mode,
                                }
                                params_json_path = write_params_json(log_dir, params)
                                # try locate metrics file
                                cand = os.path.join(log_dir, f'metrics_test_{n_exp}.txt')
                                if os.path.exists(cand):
                                    metrics_txt_path = cand
                                    status = 'done'
                                else:
                                    # wait briefly and recheck
                                    time.sleep(2)
                                    if os.path.exists(cand):
                                        metrics_txt_path = cand
                                        status = 'done'
                                    else:
                                        status = 'no_metrics'
                            else:
                                status = 'no_log_dir'
                        except Exception as e:
                            status = f'error:{type(e).__name__}'

                    with open(runs_csv, 'a', encoding='utf-8') as f:
                        f.write(','.join([
                            ts,
                            args.dataset,
                            str(n_exp),
                            str(seed),
                            str(alpha),
                            str(tau),
                            str(lvl),
                            str(args.batch_size),
                            str(args.max_epochs),
                            str(args.patience),
                            'True' if wandb_flag else 'False',
                            args.wandb_mode,
                            cmd_str.replace(',', ';'),
                            (log_dir_printed or '').replace(',', ';') if log_dir_printed else '',
                            params_json_path or '',
                            metrics_txt_path or '',
                            status,
                        ]) + '\n')

                    n_exp += 1

    print(f"Sweep completed. Runs CSV: {runs_csv}")

    if summary_flag and not dry_run_flag:
        # generate test summary at repo root
        try:
            subprocess.run([python_exe, os.path.join('.', 'collect_metrics_summary.py'), '--split', 'test'], check=False)
        except Exception:
            pass


if __name__ == '__main__':
    main()