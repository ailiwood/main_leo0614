"""
P6AB: TAV Ablation Results Analyzer

Reads all 7 ablation variant results and produces:
  1. tav_ablation_results.csv — main metrics table
  2. tav_ablation_results.md — formatted results report
  3. tav_ablation_statistics.md — paired bootstrap, McNemar, etc.
  4. tav_weight_analysis.md — AWAF weight distributions for F0

Usage:
  /e/Anaconda3/envs/mme_xlstm_stable/python.exe scripts/analyze_tav_ablation_results.py
"""
import os, sys, json, csv, glob
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_ROOT = 'outputs/P6AB_tav_ablation/mosei'
REPORT_DIR = 'reports/P6AB_tav_causal_evidence/mosei'

VARIANT_INFO = {
    'F0': {'name': 'Full TAV AWAF sLSTM', 'role': 'control', 'mods': 'T+A+V'},
    'F1': {'name': 'No Vision (T+A)', 'role': 'vision_contribution', 'mods': 'T+A'},
    'F2': {'name': 'No Audio (T+V)', 'role': 'audio_contribution', 'mods': 'T+V'},
    'F3': {'name': 'Global Static Weights', 'role': 'sample_level_awaf', 'mods': 'T+A+V'},
    'F4': {'name': 'Fixed Mean (1/3 each)', 'role': 'dynamic_fusion', 'mods': 'T+A+V'},
    'F5': {'name': 'AWAF No Interaction', 'role': 'hadamard_interaction', 'mods': 'T+A+V'},
    'E1': {'name': 'LSTM Temporal Encoder', 'role': 'slstm_contribution', 'mods': 'T+A+V'},
}

def find_latest_run(var_id, root=OUTPUT_ROOT):
    """Find the most recent completed run for a variant."""
    pattern = os.path.join(root, f'{var_id}_*')
    dirs = sorted(glob.glob(pattern), reverse=True)
    for d in dirs:
        result_path = os.path.join(d, 'result.json')
        if os.path.exists(result_path):
            return d
    return None

def load_result(run_dir):
    """Load result.json from a run directory."""
    result_path = os.path.join(run_dir, 'result.json')
    if not os.path.exists(result_path):
        return None
    with open(result_path) as f:
        return json.load(f)

def load_predictions(run_dir):
    """Load predictions CSV."""
    pred_path = os.path.join(run_dir, 'predictions_test.csv')
    if not os.path.exists(pred_path):
        return None
    rows = []
    with open(pred_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_metrics(run_dir):
    """Load metrics_epoch.csv."""
    metrics_path = os.path.join(run_dir, 'metrics_epoch.csv')
    if not os.path.exists(metrics_path):
        return None
    rows = []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def bootstrap_paired_ci(a, b, n_bootstrap=10000, alpha=0.05):
    """Paired bootstrap 95% CI for difference a - b."""
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    n = len(a)
    diffs = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        diffs[i] = (a[idx] - b[idx]).mean()
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper, diffs.mean()

def mcnemar_test(preds_a, preds_b, labels):
    """McNemar test for binary (ACC2) agreement between two models."""
    correct_a = np.array([int(float(p['final_correct'])) for p in preds_a])
    correct_b = np.array([int(float(p['final_correct'])) for p in preds_b])
    labels = np.array(labels)

    # Both correct, both wrong, a-only, b-only
    n_both_correct = np.sum((correct_a == 1) & (correct_b == 1))
    n_both_wrong = np.sum((correct_a == 0) & (correct_b == 0))
    n_a_only = np.sum((correct_a == 1) & (correct_b == 0))
    n_b_only = np.sum((correct_a == 0) & (correct_b == 1))

    # McNemar statistic
    if n_a_only + n_b_only == 0:
        return float('inf'), 1.0

    chi2 = (abs(n_a_only - n_b_only) - 1)**2 / (n_a_only + n_b_only)
    return chi2, n_a_only, n_b_only, n_both_correct, n_both_wrong

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("P6AB TAV Ablation Results Analyzer")
    print("=" * 70)

    # Load all results
    all_results = {}
    all_preds = {}
    all_metrics = {}

    for var_id in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'E1']:
        run_dir = find_latest_run(var_id)
        if run_dir:
            result = load_result(run_dir)
            preds = load_predictions(run_dir)
            metrics = load_metrics(run_dir)
            if result:
                all_results[var_id] = result
                all_preds[var_id] = preds
                all_metrics[var_id] = metrics
                print(f"  {var_id}: ACC2={result.get('final_ACC2', 'N/A'):.2f}% (epoch {result.get('best_epoch', '?')}) — {run_dir}")
            else:
                print(f"  {var_id}: NO RESULT in {run_dir}")
        else:
            print(f"  {var_id}: NO RUN FOUND")

    if not all_results:
        print("\nNo results found. Aborting.")
        return

    # ================================================================
    # 1. Results CSV
    # ================================================================
    csv_path = os.path.join(REPORT_DIR, 'tav_ablation_results.csv')
    fields = ['variant', 'name', 'role', 'modalities', 'epochs', 'best_epoch',
              'ACC2_Non0', 'F1_Non0', 'MAE', 'Corr', 'ACC7',
              'trainable_M', 'w_t', 'w_a', 'w_v', 'config']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for var_id in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'E1']:
            if var_id not in all_results:
                continue
            r = all_results[var_id]
            info = VARIANT_INFO[var_id]
            writer.writerow({
                'variant': var_id,
                'name': info['name'],
                'role': info['role'],
                'modalities': info['mods'],
                'epochs': r.get('epochs', '?'),
                'best_epoch': r.get('best_epoch', '?'),
                'ACC2_Non0': f"{r.get('final_ACC2', 0):.2f}",
                'F1_Non0': f"{r.get('F1_Non0', 0):.2f}",
                'MAE': f"{r.get('MAE', 0):.4f}",
                'Corr': f"{r.get('Corr', 0):.4f}",
                'ACC7': f"{r.get('ACC7', 0):.2f}",
                'trainable_M': r.get('trainable_M', 0),
                'w_t': f"{r.get('awaf_w_t', 0):.4f}",
                'w_a': f"{r.get('awaf_w_a', 0):.4f}",
                'w_v': f"{r.get('awaf_w_v', 0):.4f}",
                'config': r.get('config', ''),
            })
    print(f"\nResults CSV: {csv_path}")

    # ================================================================
    # 2. Paired comparisons (F0 vs each other)
    # ================================================================
    if 'F0' in all_preds and all_preds['F0']:
        f0_preds = all_preds['F0']
        f0_final = np.array([float(p['final_pred']) for p in f0_preds])
        labels = np.array([float(p['label']) for p in f0_preds])

        stat_lines = []
        stat_lines.append("# P6AB TAV Ablation Statistics\n")
        stat_lines.append(f"**Date**: 2026-06-21\n\n")
        stat_lines.append("## Paired Comparisons (F0 = Full TAV control)\n\n")

        for var_id in ['F1', 'F2', 'F3', 'F4', 'F5', 'E1']:
            if var_id not in all_preds or not all_preds[var_id]:
                continue
            info = VARIANT_INFO[var_id]
            preds_b = all_preds[var_id]
            b_final = np.array([float(p['final_pred']) for p in preds_b])

            # ACC2 bootstrap — use ACC2_Non0 (exclude label=0)
            f0_p = np.array([float(p['final_pred']) for p in f0_preds])
            b_p = np.array([float(p['final_pred']) for p in preds_b])
            mask_non0 = labels != 0.0
            f0_acc2 = np.where(f0_p[mask_non0] >= 0, 1.0, -1.0) == np.where(labels[mask_non0] >= 0, 1.0, -1.0)
            b_acc2 = np.where(b_p[mask_non0] >= 0, 1.0, -1.0) == np.where(labels[mask_non0] >= 0, 1.0, -1.0)
            f0_acc2 = f0_acc2.astype(float)
            b_acc2 = b_acc2.astype(float)
            ci_low, ci_high, mean_diff = bootstrap_paired_ci(f0_acc2, b_acc2)

            # MAE bootstrap
            f0_mae_elem = np.abs(f0_final - labels)
            b_mae_elem = np.abs(b_final - labels)
            mae_ci_low, mae_ci_high, mae_mean_diff = bootstrap_paired_ci(f0_mae_elem, b_mae_elem)

            # McNemar
            chi2, na, nb, nbc, nbw = mcnemar_test(f0_preds, preds_b, labels)

            # Prediction correlation (non-zero labels only)
            corr = np.corrcoef(f0_p[mask_non0], b_p[mask_non0])[0, 1]

            f0_acc = all_results['F0'].get('final_ACC2', 0)
            b_acc = all_results.get(var_id, {}).get('final_ACC2', 0)
            diff = f0_acc - b_acc

            stat_lines.append(f"### F0 vs {var_id}: {info['name']}\n")
            stat_lines.append(f"| Metric | F0 | {var_id} | Difference | 95% CI |\n")
            stat_lines.append(f"|--------|----|----|----|----|\n")
            stat_lines.append(f"| ACC2_Non0 | {f0_acc:.2f}% | {b_acc:.2f}% | {diff:+.2f}pp | [{ci_low*100:+.2f}, {ci_high*100:+.2f}] pp |\n")
            stat_lines.append(f"| MAE | {all_results['F0'].get('MAE',0):.4f} | {all_results.get(var_id,{}).get('MAE',0):.4f} | {mae_mean_diff:+.4f} | [{mae_ci_low:+.4f}, {mae_ci_high:+.4f}] |\n")
            stat_lines.append(f"| Pred Corr | — | — | {corr:.4f} | — |\n")
            stat_lines.append(f"\n")
            stat_lines.append(f"- McNemar: chi2={chi2:.2f}, A-only={na}, B-only={nb}\n")
            stat_lines.append(f"- Interpretation: ")

            if ci_low > 0:
                stat_lines.append(f"F0 significantly BETTER than {var_id} (CI excludes 0).\n")
            elif ci_high < 0:
                stat_lines.append(f"F0 significantly WORSE than {var_id} (CI excludes 0).\n")
            else:
                stat_lines.append(f"No significant difference (CI includes 0).\n")
            stat_lines.append(f"\n")

        # Weight analysis for F0
        stat_lines.append("## F0 AWAF Weight Analysis\n\n")
        if 'F0' in all_preds:
            f0_awaf = all_preds['F0']
            # Load awaf weights CSV
            awaf_path = os.path.join(find_latest_run('F0'), 'awaf_weights_test.csv')
            if os.path.exists(awaf_path):
                w_t, w_a, w_v = [], [], []
                with open(awaf_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        w_t.append(float(row['w_t']))
                        w_a.append(float(row['w_a']))
                        w_v.append(float(row['w_v']))
                w_t = np.array(w_t); w_a = np.array(w_a); w_v = np.array(w_v)

                stat_lines.append(f"| Stat | w_t (text) | w_a (audio) | w_v (vision) |\n")
                stat_lines.append(f"|------|-----------|------------|-------------|\n")
                stat_lines.append(f"| Mean | {w_t.mean():.4f} | {w_a.mean():.4f} | {w_v.mean():.4f} |\n")
                stat_lines.append(f"| Std | {w_t.std():.4f} | {w_a.std():.4f} | {w_v.std():.4f} |\n")
                stat_lines.append(f"| Min | {w_t.min():.4f} | {w_a.min():.4f} | {w_v.min():.4f} |\n")
                stat_lines.append(f"| Max | {w_t.max():.4f} | {w_a.max():.4f} | {w_v.max():.4f} |\n")

                # Entropy per sample
                eps = 1e-8
                entropy = -(w_t * np.log(w_t + eps) + w_a * np.log(w_a + eps) + w_v * np.log(w_v + eps))
                stat_lines.append(f"| Avg Entropy | {entropy.mean():.4f} | — | — |\n")

        stat_path = os.path.join(REPORT_DIR, 'tav_ablation_statistics.md')
        with open(stat_path, 'w') as f:
            f.writelines(stat_lines)
        print(f"Statistics: {stat_path}")

    # ================================================================
    # 3. Results markdown
    # ================================================================
    md_path = os.path.join(REPORT_DIR, 'tav_ablation_results.md')
    with open(md_path, 'w') as f:
        f.write("# P6AB MOSEI TAV Ablation Results\n\n")
        f.write(f"**Date**: 2026-06-21\n")
        f.write(f"**Protocol**: 4 epochs, seed=42, cohort=mosei_official_tav_intersection_v1\n\n")

        f.write("## Results Table\n\n")
        f.write("| Var | Description | Mods | Best Ep | ACC2 | F1 | MAE | Corr | ACC7 | Params(M) | w_t | w_a | w_v |\n")
        f.write("|-----|-------------|------|---------|------|----|-----|------|------|-----------|-----|-----|-----|\n")
        for var_id in ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'E1']:
            if var_id not in all_results:
                continue
            r = all_results[var_id]
            info = VARIANT_INFO[var_id]
            f.write(f"| {var_id} | {info['name']} | {info['mods']} | {r.get('best_epoch','?')} | "
                    f"{r.get('final_ACC2',0):.2f} | {r.get('F1_Non0',0):.2f} | "
                    f"{r.get('MAE',0):.4f} | {r.get('Corr',0):.4f} | {r.get('ACC7',0):.2f} | "
                    f"{r.get('trainable_M',0):.2f} | "
                    f"{r.get('awaf_w_t',0):.3f} | {r.get('awaf_w_a',0):.3f} | {r.get('awaf_w_v',0):.3f} |\n")

        f.write("\n## Key Comparisons\n\n")
        f.write("| Comparison | What it measures | F0 ACC2 | Variant ACC2 | Delta |\n")
        f.write("|------------|-----------------|---------|-------------|-------|\n")

        f0_acc = all_results.get('F0', {}).get('final_ACC2', 0)
        comparisons = [
            ('F0 vs F1', 'Vision modality contribution', 'F1'),
            ('F0 vs F2', 'Audio modality contribution', 'F2'),
            ('F0 vs F3', 'Sample-level dynamic weight contribution', 'F3'),
            ('F0 vs F4', 'Dynamic/learned fusion contribution', 'F4'),
            ('F0 vs F5', 'Hadamard interaction term contribution', 'F5'),
            ('F0 vs E1', 'sLSTM vs plain LSTM contribution', 'E1'),
        ]
        for comp_name, what, var_id in comparisons:
            var_acc = all_results.get(var_id, {}).get('final_ACC2', 0)
            delta = f0_acc - var_acc
            f.write(f"| {comp_name} | {what} | {f0_acc:.2f} | {var_acc:.2f} | {delta:+.2f} |\n")

    print(f"Results MD: {md_path}")

    print(f"\n{'='*70}")
    print(f"Analysis complete. Reports in {REPORT_DIR}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
