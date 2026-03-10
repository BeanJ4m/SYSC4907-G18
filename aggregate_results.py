#!/usr/bin/env python3
"""Aggregate result JSONs under the `results/` directory into a CSV summary.

Usage:
    python aggregate_results.py --results-dir results --out results/results_summary.csv
"""

import argparse
import csv
import json
import os
import re
import pickle
from datetime import timedelta

from sklearn.metrics import classification_report


def parse_folder_name(name: str):
    model = 'Unknown'
    if re.search(r'(^|[-_])TT', name, re.IGNORECASE):
        model = 'TT'
    elif re.search(r'(^|[-_])DNN', name, re.IGNORECASE):
        model = 'DNN'

    attacks = None
    m_pre = re.search(r'([0-9]+)[-_]atk', name, re.IGNORECASE)
    if m_pre:
        attacks = int(m_pre.group(1))
    else:
        m_post = re.search(r'[-_]atk[-_]?([0-9]+)', name, re.IGNORECASE)
        if m_post:
            attacks = int(m_post.group(1))

    return model, attacks


def hhmmss_from_human(s: str):
    if not s:
        return ''
    s = s.strip()
    hours = minutes = seconds = 0

    mh = re.search(r'(\d+)h', s)
    mm = re.search(r'(\d+)m', s)
    ms = re.search(r'(\d+)s', s)

    if mh:
        hours = int(mh.group(1))
    if mm:
        minutes = int(mm.group(1))
    if ms:
        seconds = int(ms.group(1))

    return str(timedelta(hours=hours, minutes=minutes, seconds=seconds))


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def flatten_labels(obj):
    """
    Convert loaded label data into a flat list of strings suitable for sklearn metrics.
    Handles lists, tuples, numpy-like arrays, pandas Series, and line-based text.
    """
    if obj is None:
        return []

    # pandas / numpy style
    if hasattr(obj, "tolist"):
        obj = obj.tolist()

    # single scalar
    if isinstance(obj, (str, int, float, bool)):
        return [str(obj).strip()]

    # list / tuple / nested
    if isinstance(obj, (list, tuple)):
        flat = []
        for item in obj:
            if hasattr(item, "tolist"):
                item = item.tolist()

            if isinstance(item, (list, tuple)):
                flat.extend(flatten_labels(item))
            else:
                flat.append(str(item).strip())
        return flat

    # fallback
    return [str(obj).strip()]


def load_label_file(path):
    """
    Robust loader for actual/pred files.
    Tries pickle first, then text encodings.
    """
    # Try pickle first because 0x80 at byte 0 usually means pickle protocol.
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return flatten_labels(obj)
    except Exception:
        pass

    # Try UTF-8 text
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() != '']
    except Exception:
        pass

    # Try latin-1 text as final fallback
    try:
        with open(path, 'r', encoding='latin-1') as f:
            return [line.strip() for line in f if line.strip() != '']
    except Exception:
        pass

    return []


def aggregate_folder(path):
    name = os.path.basename(path.rstrip('/'))
    model, attacks = parse_folder_name(name)

    pkl_path = os.path.join(path, 'PD_Results.pkl')
    results_data = None

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results_data = pickle.load(f)

        usecase = list(results_data.keys())[0]
        accs = results_data[usecase].get('Accuracy', [])
        round_nums = list(range(1, len(accs) + 1))
    else:
        rounds = []
        for fn in os.listdir(path):
            if fn.startswith('Round_') and fn.endswith('_Summary.json'):
                rounds.append(os.path.join(path, fn))
        rounds.sort()

        accs = []
        round_nums = []
        for rfile in rounds:
            data = read_json(rfile)
            if not data:
                continue
            round_nums.append(data.get('round'))
            if 'accuracy' in data:
                accs.append(float(data['accuracy']))

    macro_f1s = []
    macro_precs = []
    macro_recs = []
    weighted_f1s = []
    weighted_precs = []
    weighted_recs = []

    if os.path.exists(pkl_path) and results_data is not None:
        usecase = list(results_data.keys())[0]
        weighted_f1s = results_data[usecase].get('F1_Score', [])
        weighted_precs = results_data[usecase].get('Precision', [])
        weighted_recs = results_data[usecase].get('Recall', [])

        for r in round_nums:
            actual_file = os.path.join(path, f'Global_{r}_actual')
            pred_file = os.path.join(path, f'Global_{r}_pred')

            if os.path.exists(actual_file) and os.path.exists(pred_file):
                y_true = load_label_file(actual_file)
                y_pred = load_label_file(pred_file)

                if y_true and y_pred and len(y_true) == len(y_pred):
                    report = classification_report(
                        y_true,
                        y_pred,
                        output_dict=True,
                        zero_division=0
                    )
                    macro_f1s.append(report['macro avg']['f1-score'])
                    macro_precs.append(report['macro avg']['precision'])
                    macro_recs.append(report['macro avg']['recall'])
                else:
                    macro_f1s.append('')
                    macro_precs.append('')
                    macro_recs.append('')
            else:
                macro_f1s.append('')
                macro_precs.append('')
                macro_recs.append('')
    else:
        rounds = []
        for fn in os.listdir(path):
            if fn.startswith('Round_') and fn.endswith('_Summary.json'):
                rounds.append(os.path.join(path, fn))
        rounds.sort()

        for rfile in rounds:
            data = read_json(rfile)
            if not data:
                continue

            sm = data.get('summary_metrics', {})
            macro = sm.get('macro_avg', {}) if sm else {}
            weighted = sm.get('weighted_avg', {}) if sm else {}

            if macro:
                macro_f1s.append(float(macro.get('f1', '')) if 'f1' in macro else '')
                macro_precs.append(float(macro.get('precision', '')) if 'precision' in macro else '')
                macro_recs.append(float(macro.get('recall', '')) if 'recall' in macro else '')

            if weighted:
                weighted_f1s.append(float(weighted.get('f1', '')) if 'f1' in weighted else '')
                weighted_precs.append(float(weighted.get('precision', '')) if 'precision' in weighted else '')
                weighted_recs.append(float(weighted.get('recall', '')) if 'recall' in weighted else '')

    sys_path = os.path.join(path, 'system_resources.json')
    sysdata = read_json(sys_path) if os.path.exists(sys_path) else None

    fl_mode = None
    llm_mode = None
    if sysdata and isinstance(sysdata, dict):
        fl_mode = sysdata.get('fl_mode')
        llm_mode = sysdata.get('llm_mode')

    fl_str = 'Yes' if fl_mode else 'No' if fl_mode is not None else ''
    llm_str = 'Yes' if llm_mode else 'No' if llm_mode is not None else ''

    model_size_kb = None
    try:
        sizes = []
        for fn in os.listdir(path):
            if fn.startswith('GlobalModel') and fn.endswith('.pth'):
                sizes.append(os.path.getsize(os.path.join(path, fn)))
        if sizes:
            model_size_kb = max(sizes) / 1024.0
    except Exception:
        model_size_kb = None

    plateau = ''
    if len(accs) >= 3:
        tol = 2.0  # percentage points
        min_stable_len = 4

        for start in range(len(accs) - min_stable_len + 1):
            tail = accs[start:]
            stable = True

            for j in range(len(tail) - 1):
                if abs(tail[j + 1] - tail[j]) > tol:
                    stable = False
                    break

            if stable:
                plateau = round_nums[start]

    def stats(arr):
        clean = [x for x in arr if isinstance(x, (int, float))]
        if not clean:
            return ('', '', '')
        return (min(clean), max(clean), sum(clean) / len(clean))

    results = []
    result_base = {
        'Attacks': attacks,
        'Model': model,
        'FL_Mode': fl_str,
        'LLM_Mode': llm_str,
    }

    a_min, a_max, a_avg = stats(accs)
    final_acc = accs[-1] if accs else ''
    results.append({
        **result_base,
        'Metric': 'Accuracy',
        'Min': a_min,
        'Max': a_max,
        'Average': a_avg,
        'Final': final_acc
    })

    if plateau:
        results.append({
            **result_base,
            'Metric': 'Plateau_Round',
            'Min': '',
            'Max': '',
            'Average': '',
            'Final': plateau
        })

    mf_min, mf_max, mf_avg = stats(macro_f1s)
    results.append({
        **result_base,
        'Metric': 'Macro_F1',
        'Min': mf_min,
        'Max': mf_max,
        'Average': mf_avg,
        'Final': (macro_f1s[-1] if macro_f1s else '')
    })

    mp_min, mp_max, mp_avg = stats(macro_precs)
    results.append({
        **result_base,
        'Metric': 'Macro_Precision',
        'Min': mp_min,
        'Max': mp_max,
        'Average': mp_avg,
        'Final': (macro_precs[-1] if macro_precs else '')
    })

    mr_min, mr_max, mr_avg = stats(macro_recs)
    results.append({
        **result_base,
        'Metric': 'Macro_Recall',
        'Min': mr_min,
        'Max': mr_max,
        'Average': mr_avg,
        'Final': (macro_recs[-1] if macro_recs else '')
    })

    wf_min, wf_max, wf_avg = stats(weighted_f1s)
    results.append({
        **result_base,
        'Metric': 'Weighted_F1',
        'Min': wf_min,
        'Max': wf_max,
        'Average': wf_avg,
        'Final': (weighted_f1s[-1] if weighted_f1s else '')
    })

    wp_min, wp_max, wp_avg = stats(weighted_precs)
    results.append({
        **result_base,
        'Metric': 'Weighted_Precision',
        'Min': wp_min,
        'Max': wp_max,
        'Average': wp_avg,
        'Final': (weighted_precs[-1] if weighted_precs else '')
    })

    wr_min, wr_max, wr_avg = stats(weighted_recs)
    results.append({
        **result_base,
        'Metric': 'Weighted_Recall',
        'Min': wr_min,
        'Max': wr_max,
        'Average': wr_avg,
        'Final': (weighted_recs[-1] if weighted_recs else '')
    })

    if sysdata:
        sr = sysdata.get('system_resources', {})

        cpu = sr.get('cpu_percent', {})
        if cpu:
            results.append({
                **result_base,
                'Metric': 'CPU_Train_Util_%',
                'Min': cpu.get('min', ''),
                'Max': cpu.get('max', ''),
                'Average': cpu.get('avg', ''),
                'Final': ''
            })

        ram = sr.get('ram_mb', {})
        if ram:
            results.append({
                **result_base,
                'Metric': 'RAM_Train_MB',
                'Min': ram.get('min', ''),
                'Max': ram.get('max', ''),
                'Average': ram.get('avg', ''),
                'Final': ''
            })

        gpu = sr.get('gpu_util', {})
        if gpu:
            results.append({
                **result_base,
                'Metric': 'GPU_Util_%',
                'Min': gpu.get('min', ''),
                'Max': gpu.get('max', ''),
                'Average': gpu.get('avg', ''),
                'Final': ''
            })

        gpupower = sr.get('gpu_power', {}) or sr.get('gpu_power_w', {})
        if gpupower:
            results.append({
                **result_base,
                'Metric': 'GPU_Power_W',
                'Min': gpupower.get('min', ''),
                'Max': gpupower.get('max', ''),
                'Average': gpupower.get('avg', ''),
                'Final': ''
            })

        gpumem = sr.get('gpu_mem_mb', {}) or sr.get('gpu_mem_util', {})
        if gpumem:
            results.append({
                **result_base,
                'Metric': 'GPU_VRAM_MB',
                'Min': gpumem.get('min', ''),
                'Max': gpumem.get('max', ''),
                'Average': gpumem.get('avg', ''),
                'Final': ''
            })

        tstr = ''
        if isinstance(sysdata, dict) and 'time' in sysdata:
            tstr = hhmmss_from_human(sysdata.get('time', ''))

        if not tstr:
            gp_files = [
                os.path.join(path, fn)
                for fn in os.listdir(path)
                if fn.startswith('GlobalModel') and fn.endswith('.pth')
            ]
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
                **result_base,
                'Metric': 'Train_Time',
                'Min': '',
                'Max': '',
                'Average': '',
                'Final': tstr
            })

    if model_size_kb is not None:
        results.append({
            **result_base,
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

    fieldnames = ['Attacks', 'Model', 'FL_Mode', 'LLM_Mode', 'Metric', 'Min', 'Max', 'Average', 'Final']
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {k: ('' if v is None else v) for k, v in r.items()}
            writer.writerow(out)

    print('Wrote', args.out)


if __name__ == '__main__':
    main()