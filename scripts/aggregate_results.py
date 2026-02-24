#!/usr/bin/env python3
"""Aggregate result JSONs under the `results/` directory into a CSV summary.

Usage: python scripts/aggregate_results.py --results-dir results --out results/results_summary.csv
"""
import argparse
import csv
import json
import os
import re
from datetime import timedelta


def parse_folder_name(name: str):
    # detect model type (TT or DNN) and number of attacks
    model = 'Unknown'
    if re.search(r'(^|[-_])TT', name, re.IGNORECASE):
        model = 'TT'
    elif re.search(r'(^|[-_])DNN', name, re.IGNORECASE):
        model = 'DNN'

    attacks = None
    # try number before 'atk' first (e.g. '...-5-atk-...')
    m_pre = re.search(r'([0-9]+)[-_]atk', name, re.IGNORECASE)
    if m_pre:
        attacks = int(m_pre.group(1))
    else:
        # fallback to number after 'atk' (e.g. '...-atk-5-...')
        m_post = re.search(r'[-_]atk[-_]?([0-9]+)', name, re.IGNORECASE)
        if m_post:
            attacks = int(m_post.group(1))

    return model, attacks


def hhmmss_from_human(s: str):
    # convert strings like '47m 42s' or '1h 2m 3s' to HH:MM:SS
    if not s:
        return ''
    s = s.strip()
    hours = minutes = seconds = 0
    mh = re.search(r"(\d+)h", s)
    mm = re.search(r"(\d+)m", s)
    ms = re.search(r"(\d+)s", s)
    if mh:
        hours = int(mh.group(1))
    if mm:
        minutes = int(mm.group(1))
    if ms:
        seconds = int(ms.group(1))
    return str(timedelta(hours=hours, minutes=minutes, seconds=seconds))


def read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def aggregate_folder(path):
    name = os.path.basename(path.rstrip('/'))
    model, attacks = parse_folder_name(name)

    # find round summary files
    rounds = []
    for fn in os.listdir(path):
        if fn.startswith('Round_') and fn.endswith('_Summary.json'):
            rounds.append(os.path.join(path, fn))
    rounds.sort()

    accs = []
    macro_f1s = []
    macro_precs = []
    macro_recs = []
    weighted_f1s = []
    weighted_precs = []
    weighted_recs = []
    round_nums = []
    for rfile in rounds:
        data = read_json(rfile)
        if not data:
            continue
        round_nums.append(data.get('round'))
        # accuracy top-level
        if 'accuracy' in data:
            accs.append(float(data['accuracy']))
        # try to extract macro avg metrics
        sm = data.get('summary_metrics', {})
        macro = sm.get('macro_avg', {}) if sm else {}
        weighted = sm.get('weighted_avg', {}) if sm else {}
        if macro:
            if 'f1' in macro:
                macro_f1s.append(float(macro['f1']))
            if 'precision' in macro:
                macro_precs.append(float(macro['precision']))
            if 'recall' in macro:
                macro_recs.append(float(macro['recall']))
        if weighted:
            if 'f1' in weighted:
                weighted_f1s.append(float(weighted['f1']))
            if 'precision' in weighted:
                weighted_precs.append(float(weighted['precision']))
            if 'recall' in weighted:
                weighted_recs.append(float(weighted['recall']))

    # system resources
    sys_path = os.path.join(path, 'system_resources.json')
    sysdata = read_json(sys_path) if os.path.exists(sys_path) else None

    # model size: pick the largest GlobalModel_*.pth if present
    model_size_kb = None
    try:
        sizes = []
        for fn in os.listdir(path):
            if fn.startswith('GlobalModel') and fn.endswith('.pth'):
                sizes.append(os.path.getsize(os.path.join(path, fn)))
        if sizes:
            # use largest (final) model
            model_size_kb = max(sizes) / 1024.0
    except Exception:
        model_size_kb = None

    # compute plateau round for accuracy: find first round after which change is small
    plateau = ''
    if len(accs) >= 3:
        tol = 1e-3
        for i in range(len(accs)-2):
            if abs(accs[i+1] - accs[i]) <= tol and abs(accs[i+2] - accs[i+1]) <= tol:
                plateau = round_nums[i+1] if i+1 < len(round_nums) else ''
                break

    def stats(arr):
        if not arr:
            return ('', '', '')
        return (min(arr), max(arr), sum(arr)/len(arr))

    results = []

    # Accuracy
    a_min, a_max, a_avg = stats(accs)
    final_acc = accs[-1] if accs else ''
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Accuracy',
        'Min': a_min,
        'Max': a_max,
        'Average': a_avg,
        'Final': final_acc
    })
    
    # Plateau Round (as a metric)
    if plateau:
        results.append({
            'Attacks': attacks,
            'Model': model,
            'Metric': 'Plateau_Round',
            'Min': '',
            'Max': '',
            'Average': '',
            'Final': plateau
        })

    # Macro averages: F1, Precision, Recall
    mf_min, mf_max, mf_avg = stats(macro_f1s)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Macro_F1',
        'Min': mf_min,
        'Max': mf_max,
        'Average': mf_avg,
        'Final': (macro_f1s[-1] if macro_f1s else '')
    })
    mp_min, mp_max, mp_avg = stats(macro_precs)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Macro_Precision',
        'Min': mp_min,
        'Max': mp_max,
        'Average': mp_avg,
        'Final': (macro_precs[-1] if macro_precs else '')
    })
    mr_min, mr_max, mr_avg = stats(macro_recs)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Macro_Recall',
        'Min': mr_min,
        'Max': mr_max,
        'Average': mr_avg,
        'Final': (macro_recs[-1] if macro_recs else '')
    })

    # Weighted averages: F1, Precision, Recall
    wf_min, wf_max, wf_avg = stats(weighted_f1s)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Weighted_F1',
        'Min': wf_min,
        'Max': wf_max,
        'Average': wf_avg,
        'Final': (weighted_f1s[-1] if weighted_f1s else '')
    })
    wp_min, wp_max, wp_avg = stats(weighted_precs)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Weighted_Precision',
        'Min': wp_min,
        'Max': wp_max,
        'Average': wp_avg,
        'Final': (weighted_precs[-1] if weighted_precs else '')
    })
    wr_min, wr_max, wr_avg = stats(weighted_recs)
    results.append({
        'Attacks': attacks,
        'Model': model,
        'Metric': 'Weighted_Recall',
        'Min': wr_min,
        'Max': wr_max,
        'Average': wr_avg,
        'Final': (weighted_recs[-1] if weighted_recs else '')
    })

    # system resources metrics
    if sysdata:
        sr = sysdata.get('system_resources', {})
        cpu = sr.get('cpu_percent', {})
        if cpu:
            results.append({
                    'Attacks': attacks,
                    'Model': model,
                    'Metric': 'CPU_Train_Util_%',
                    'Min': cpu.get('min',''),
                    'Max': cpu.get('max',''),
                    'Average': cpu.get('avg',''),
                    'Final': ''
                })

        ram = sr.get('ram_mb', {})
        if ram:
            results.append({
                'Attacks': attacks,
                'Model': model,
                'Metric': 'RAM_Train_MB',
                'Min': ram.get('min',''),
                'Max': ram.get('max',''),
                'Average': ram.get('avg',''),
                'Final': ''
            })

        gpu = sr.get('gpu_util', {})
        if gpu:
            results.append({
                'Attacks': attacks,
                'Model': model,
                'Metric': 'GPU_Util_%',
                'Min': gpu.get('min',''),
                'Max': gpu.get('max',''),
                'Average': gpu.get('avg',''),
                'Final': ''
            })

        gpupower = sr.get('gpu_power_w', {})
        if gpupower:
            results.append({
                'Attacks': attacks,
                'Model': model,
                'Metric': 'GPU_Power_W',
                'Min': gpupower.get('min',''),
                'Max': gpupower.get('max',''),
                'Average': gpupower.get('avg',''),
                'Final': ''
            })

        gpumem = sr.get('gpu_mem_mb', {}) or sr.get('gpu_mem_util', {})
        if gpumem:
            # gpu_mem_mb uses min/max/avg in MB
            results.append({
                'Attacks': attacks,
                'Model': model,
                'Metric': 'GPU_VRAM_MB',
                'Min': gpumem.get('min',''),
                'Max': gpumem.get('max',''),
                'Average': gpumem.get('avg',''),
                'Final': ''
            })

        # Train time: prefer explicit 'time' field; else try to approximate
        tstr = ''
        if isinstance(sysdata, dict) and 'time' in sysdata:
            tstr = hhmmss_from_human(sysdata.get('time',''))
        # fallback: if we have many GlobalModel_*.pth files, use their mtimes
        if not tstr:
            gp_files = [os.path.join(path, fn) for fn in os.listdir(path) if fn.startswith('GlobalModel') and fn.endswith('.pth')]
            if gp_files:
                try:
                    mtimes = [os.path.getmtime(f) for f in gp_files]
                    span = max(mtimes) - min(mtimes)
                    if span > 0:
                        tstr = str(timedelta(seconds=int(span)))
                except Exception:
                    tstr = ''

        if tstr:
            results.append({
                'Attacks': attacks,
                'Model': model,
                'Metric': 'Train_Time',
                'Min': '',
                'Max': '',
                'Average': '',
                'Final': tstr
            })

    # Model size
    if model_size_kb is not None:
        results.append({
            'Attacks': attacks,
            'Model': model,
            'Metric': 'Model_Size_KB',
            'Min': '',
            'Max': '',
            'Average': '',
            'Final': round(model_size_kb, 1)
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', help='Path to results directory')
    parser.add_argument('--out', default='results/results_summary.csv', help='Output CSV path')
    args = parser.parse_args()

    rows = []
    if not os.path.isdir(args.results_dir):
        print('results directory not found:', args.results_dir)
        return

    for entry in sorted(os.listdir(args.results_dir)):
        folder = os.path.join(args.results_dir, entry)
        if not os.path.isdir(folder):
            continue
        rows.extend(aggregate_folder(folder))

    # write CSV (no separate Time column; time is a metric row)
    fieldnames = ['Attacks','Model','Metric','Min','Max','Average','Final']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # normalize values
            out = {k: ('' if v is None else v) for k,v in r.items()}
            writer.writerow(out)

    print('Wrote', args.out)


if __name__ == '__main__':
    main()
