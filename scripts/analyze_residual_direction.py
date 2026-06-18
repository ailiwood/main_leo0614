#!/usr/bin/env python
"""P6J: Residual direction audit — analyze whether delta corrects text_base errors."""
import sys, os, csv, json
import numpy as np

def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def main():
    pred_csv = sys.argv[1] if len(sys.argv) > 1 else 'outputs/P6I/text_conf_residual_s42_20260618_233629/predictions_test.csv'
    tconf_csv = pred_csv.replace('predictions_test.csv', 'text_confidence_test.csv')
    out_dir = os.path.dirname(pred_csv)

    rows = load_csv(pred_csv)
    tconf_rows = {}
    if os.path.exists(tconf_csv):
        for r in load_csv(tconf_csv):
            tconf_rows[r['sample_id']] = r

    # Parse
    labels = np.array([float(r['label']) for r in rows])
    tb = np.array([float(r['text_base_pred']) for r in rows])
    fn = np.array([float(r['final_pred']) for r in rows])
    delta = np.array([float(r['delta']) for r in rows])
    gate = np.array([float(r['gate']) for r in rows])
    groups = [r['group'] for r in rows]
    tb_correct = np.array([int(r['text_base_correct']) for r in rows])
    fn_correct = np.array([int(r['final_correct']) for r in rows])

    # Derived
    target_delta = labels - tb          # what residual SHOULD fix
    pred_delta = fn - tb                # what residual ACTUALLY did (= gate*delta)
    abs_error_tb = np.abs(labels - tb)
    abs_error_fn = np.abs(labels - fn)

    # Key metrics
    sign_match = (np.sign(pred_delta) == np.sign(target_delta))
    # Handle zero cases safely
    zero_mask = np.abs(target_delta) < 1e-6
    if zero_mask.any():
        sign_match[zero_mask] = (np.abs(pred_delta[zero_mask]) < 1e-6)
    delta_sign_correct_rate = sign_match.mean()

    error_reduced = abs_error_fn < abs_error_tb
    error_reduced_rate = error_reduced.mean()

    text_wrong = ~tb_correct
    final_right = fn_correct
    text_wrong_final_right = (text_wrong & final_right).sum()
    text_right = tb_correct
    final_wrong = ~fn_correct
    text_right_final_wrong = (text_right & final_wrong).sum()
    net_correct_gain = text_wrong_final_right - text_right_final_wrong

    # Per-group analysis
    group_names = ['strong_neg', 'weak_neg', 'near_zero', 'weak_pos', 'strong_pos']
    group_stats = {}
    for g in group_names:
        mask = np.array([gg == g for gg in groups])
        n = mask.sum()
        if n == 0:
            group_stats[g] = {'n': 0}
            continue
        twfr = (text_wrong[mask] & final_right[mask]).sum()
        trfw = (text_right[mask] & final_wrong[mask]).sum()
        group_stats[g] = {
            'n': int(n),
            'tb_acc': tb_correct[mask].mean() * 100,
            'fn_acc': fn_correct[mask].mean() * 100,
            'text_wrong_final_right': int(twfr),
            'text_right_final_wrong': int(trfw),
            'net_gain': int(twfr - trfw),
            'sign_correct_rate': sign_match[mask].mean() * 100,
            'error_reduced_rate': error_reduced[mask].mean() * 100,
            'mean_gate': gate[mask].mean(),
            'mean_delta': delta[mask].mean(),
            'mean_abs_error_tb': abs_error_tb[mask].mean(),
            'mean_abs_error_fn': abs_error_fn[mask].mean(),
        }

    # Gate vs error_reduced
    gate_bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    gate_stats = {}
    for lo, hi in gate_bins:
        mask = (gate >= lo) & (gate < hi)
        if mask.sum() == 0:
            continue
        gate_stats[f'{lo}-{hi}'] = {
            'n': int(mask.sum()),
            'error_reduced_rate': error_reduced[mask].mean() * 100,
            'sign_correct_rate': sign_match[mask].mean() * 100,
        }

    # Text confidence vs error_reduced
    if tconf_rows:
        tconfs = np.array([float(tconf_rows.get(r['sample_id'], {}).get('text_confidence', 0)) for r in rows])
        conf_bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 1.0)]
        conf_stats = {}
        for lo, hi in conf_bins:
            mask = (tconfs >= lo) & (tconfs < hi)
            if mask.sum() == 0:
                continue
            conf_stats[f'{lo}-{hi}'] = {
                'n': int(mask.sum()),
                'error_reduced_rate': error_reduced[mask].mean() * 100,
                'sign_correct_rate': sign_match[mask].mean() * 100,
                'gate_mean': gate[mask].mean(),
            }
    else:
        conf_stats = {}

    # Print summary
    print('=' * 70)
    print('P6J Residual Direction Audit — P6I text_confidence_residual')
    print('=' * 70)
    print(f'\nGlobal:')
    print(f'  Total samples:          {len(labels)}')
    print(f'  delta_sign_correct_rate: {delta_sign_correct_rate*100:.2f}%')
    print(f'  error_reduced_rate:      {error_reduced_rate*100:.2f}%')
    print(f'  text_wrong_final_right:  {text_wrong_final_right}')
    print(f'  text_right_final_wrong:  {text_right_final_wrong}')
    print(f'  net_correct_gain:        {net_correct_gain:+d}')
    print(f'  text_base ACC2:          {tb_correct.mean()*100:.2f}%')
    print(f'  final ACC2:              {fn_correct.mean()*100:.2f}%')
    print(f'  delta_abs_mean:          {np.abs(delta).mean():.4f}')
    print(f'  pred_delta_abs_mean:     {np.abs(pred_delta).mean():.4f}')
    print(f'  target_delta_abs_mean:   {np.abs(target_delta).mean():.4f}')
    print(f'  gate_mean:               {gate.mean():.4f}')
    print(f'  mean_abs_error_tb:       {abs_error_tb.mean():.4f}')
    print(f'  mean_abs_error_fn:       {abs_error_fn.mean():.4f}')

    print(f'\nPer-group:')
    print(f'  {"Group":<14s} {"N":>4s} {"tb_acc":>7s} {"fn_acc":>7s} {"TWFR":>5s} {"TRFW":>5s} {"net":>5s} {"sign%":>7s} {"err↓%":>7s}')
    print(f'  {"-"*70}')
    for g in group_names:
        s = group_stats[g]
        if s['n'] == 0:
            continue
        print(f'  {g:<14s} {s["n"]:4d} {s["tb_acc"]:6.2f}% {s["fn_acc"]:6.2f}% {s["text_wrong_final_right"]:5d} {s["text_right_final_wrong"]:5d} {s["net_gain"]:+5d} {s["sign_correct_rate"]:6.2f}% {s["error_reduced_rate"]:6.2f}%')

    print(f'\nGate bins vs error_reduced:')
    for k, v in gate_stats.items():
        print(f'  gate {k}: n={v["n"]:4d}  error_reduced={v["error_reduced_rate"]:.1f}%  sign_correct={v["sign_correct_rate"]:.1f}%')

    if conf_stats:
        print(f'\nText confidence bins vs error_reduced:')
        for k, v in conf_stats.items():
            print(f'  conf {k}: n={v["n"]:4d}  error_reduced={v["error_reduced_rate"]:.1f}%  gate={v["gate_mean"]:.3f}')

    # Save enriched CSV
    enriched_path = os.path.join(out_dir, 'residual_direction_analysis.csv')
    with open(enriched_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['sample_id', 'label', 'text_base_pred', 'final_pred', 'delta', 'gate',
                  'target_delta', 'pred_delta', 'delta_sign_correct', 'text_base_correct',
                  'final_correct', 'text_wrong_final_right', 'text_right_final_wrong',
                  'abs_error_tb', 'abs_error_fn', 'error_reduced', 'group']
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([
                r['sample_id'], labels[i], tb[i], fn[i], delta[i], gate[i],
                target_delta[i], pred_delta[i], int(sign_match[i]),
                int(tb_correct[i]), int(fn_correct[i]),
                int(text_wrong[i] and final_right[i]),
                int(text_right[i] and final_wrong[i]),
                abs_error_tb[i], abs_error_fn[i], int(error_reduced[i]),
                groups[i],
            ])
    print(f'\nEnriched CSV saved: {enriched_path}')

    # Save summary JSON
    summary = {
        'delta_sign_correct_rate': float(delta_sign_correct_rate),
        'error_reduced_rate': float(error_reduced_rate),
        'text_wrong_final_right': int(text_wrong_final_right),
        'text_right_final_wrong': int(text_right_final_wrong),
        'net_correct_gain': int(net_correct_gain),
        'text_base_ACC2': float(tb_correct.mean()),
        'final_ACC2': float(fn_correct.mean()),
        'delta_abs_mean': float(np.abs(delta).mean()),
        'pred_delta_abs_mean': float(np.abs(pred_delta).mean()),
        'target_delta_abs_mean': float(np.abs(target_delta).mean()),
        'gate_mean': float(gate.mean()),
        'tb_mae': float(abs_error_tb.mean()),
        'fn_mae': float(abs_error_fn.mean()),
        'group_stats': {g: {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in s.items()}
                        for g, s in group_stats.items()},
        'gate_bins': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()} for k, v in gate_stats.items()},
    }
    json.dump(summary, open(os.path.join(out_dir, 'residual_direction_summary.json'), 'w'), indent=2)
    print(f'Summary JSON saved.')


if __name__ == '__main__':
    main()
