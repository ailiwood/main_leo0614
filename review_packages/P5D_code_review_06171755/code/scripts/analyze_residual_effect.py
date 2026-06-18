#!/usr/bin/env python
"""
scripts/analyze_residual_effect.py — P5D Residual Effect Analysis

分析 text_base vs final 预测差异，回答:
  1. residual 主要修正了哪些样本
  2. residual 主要破坏了哪些样本
  3. weak_neg 是否被改善
  4. near_zero 是否被破坏
  5. strong_pos/strong_neg 是否被过度修正
  6. audio 权重高是否真的带来有效修正
  7. delta_reg 的幅度是否过大
  8. delta_scale 是否学到合理范围

Usage:
    python scripts/analyze_residual_effect.py --csv_path PATH_TO_text_base_delta_test.csv
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure required columns
    required = ['label', 'reg_text_base', 'reg_final']
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    return df


def classify_samples(df):
    """Classify samples into groups."""
    lbl = df['label'].values
    df['group'] = 'other'
    df.loc[lbl >= 1.0, 'group'] = 'strong_pos'
    df.loc[lbl <= -1.0, 'group'] = 'strong_neg'
    df.loc[(lbl > -1.0) & (lbl < 0.0), 'group'] = 'weak_neg'
    df.loc[(lbl > 0.0) & (lbl < 1.0), 'group'] = 'weak_pos'
    df.loc[np.abs(lbl) <= 0.5, 'group'] = 'near_zero'
    return df


def compute_error(df):
    """Compute per-sample errors."""
    lbl = df['label'].values
    reg_base = df['reg_text_base'].values
    reg_final = df['reg_final'].values

    df['error_text_base'] = np.abs(reg_base - lbl)
    df['error_final'] = np.abs(reg_final - lbl)
    df['abs_error_diff'] = df['error_final'] - df['error_text_base']  # negative = improved

    # Improved: final error < base error (excluding equal)
    df['improved_by_residual'] = (df['error_final'] < df['error_text_base']) & (~np.isclose(df['error_final'], df['error_text_base']))
    df['damaged_by_residual'] = (df['error_final'] > df['error_text_base'])

    # Sign correctness
    df['text_sign_correct'] = (np.sign(reg_base) == np.sign(lbl)) | (lbl == 0)
    df['final_sign_correct'] = (np.sign(reg_final) == np.sign(lbl)) | (lbl == 0)

    # Confidence proxies
    df['abs_text_confidence'] = np.abs(reg_base)
    df['abs_delta_magnitude'] = np.abs(df.get('delta_reg', 0))

    return df


def analyze(df):
    """Generate summary statistics by group."""
    groups = ['strong_pos', 'strong_neg', 'weak_pos', 'weak_neg', 'near_zero', 'all']
    summary = []

    for grp in groups:
        if grp == 'all':
            subset = df
        else:
            subset = df[df['group'] == grp]

        if len(subset) == 0:
            continue

        n = len(subset)
        n_improved = subset['improved_by_residual'].sum()
        n_damaged = subset['damaged_by_residual'].sum()
        pct_improved = 100.0 * n_improved / n
        pct_damaged = 100.0 * n_damaged / n

        mean_error_base = subset['error_text_base'].mean()
        mean_error_final = subset['error_final'].mean()
        mean_delta = subset['abs_delta_magnitude'].mean()

        text_sign_acc = subset['text_sign_correct'].mean() * 100
        final_sign_acc = subset['final_sign_correct'].mean() * 100

        summary.append({
            'group': grp,
            'n': n,
            'mean_error_text_base': mean_error_base,
            'mean_error_final': mean_error_final,
            'mean_abs_delta': mean_delta,
            'n_improved': n_improved,
            'n_damaged': n_damaged,
            'pct_improved': pct_improved,
            'pct_damaged': pct_damaged,
            'text_sign_accuracy': text_sign_acc,
            'final_sign_accuracy': final_sign_acc,
        })

        # AWAF weight breakdown by group
        if 'awaf_w_t' in subset.columns:
            summary[-1]['awaf_w_t_mean'] = subset['awaf_w_t'].mean()
            summary[-1]['awaf_w_a_mean'] = subset['awaf_w_a'].mean()
            summary[-1]['awaf_w_v_mean'] = subset['awaf_w_v'].mean()

    return pd.DataFrame(summary)


def analyze_awaf_weight_effectiveness(df):
    """Check whether high audio weight correlates with better residual correction."""
    if 'awaf_w_a' not in df.columns:
        return None

    # Split by audio weight percentile
    w_a = df['awaf_w_a'].values
    median_w_a = np.median(w_a)

    high_audio = df[w_a >= median_w_a]
    low_audio = df[w_a < median_w_a]

    results = {
        'high_audio_w_a': {
            'mean_w_a': high_audio['awaf_w_a'].mean(),
            'pct_improved': high_audio['improved_by_residual'].mean() * 100,
            'pct_damaged': high_audio['damaged_by_residual'].mean() * 100,
            'mean_error_reduction': (high_audio['error_text_base'] - high_audio['error_final']).mean(),
        },
        'low_audio_w_a': {
            'mean_w_a': low_audio['awaf_w_a'].mean(),
            'pct_improved': low_audio['improved_by_residual'].mean() * 100,
            'pct_damaged': low_audio['damaged_by_residual'].mean() * 100,
            'mean_error_reduction': (low_audio['error_text_base'] - low_audio['error_final']).mean(),
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze residual effect')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to text_base_delta_test.csv')
    parser.add_argument('--output_dir', type=str, default='reports/P5D_residual_sprint')
    parser.add_argument('--run_name', type=str, default='P5C_s42')
    args = parser.parse_args()

    print(f"Loading: {args.csv_path}")
    df = load_data(args.csv_path)
    print(f"  Samples: {len(df)}")

    df = classify_samples(df)
    df = compute_error(df)

    # Group summary
    summary = analyze(df)
    print("\n=== Group Summary ===")
    print(summary.to_string(index=False))

    # AWAF effectiveness
    awaf_eff = analyze_awaf_weight_effectiveness(df)
    if awaf_eff:
        print("\n=== AWAF Audio Weight Effectiveness ===")
        for k, v in awaf_eff.items():
            print(f"  {k}: pct_improved={v['pct_improved']:.1f}%, "
                  f"pct_damaged={v['pct_damaged']:.1f}%, "
                  f"mean_err_reduction={v['mean_error_reduction']:.4f}")

    # Overall stats
    n_total = len(df)
    n_improved = df['improved_by_residual'].sum()
    n_damaged = df['damaged_by_residual'].sum()
    n_neutral = n_total - n_improved - n_damaged

    print(f"\n=== Overall Residual Effect ===")
    print(f"Total samples: {n_total}")
    print(f"Improved: {n_improved} ({100*n_improved/n_total:.1f}%)")
    print(f"Damaged:  {n_damaged} ({100*n_damaged/n_total:.1f}%)")
    print(f"Neutral:  {n_neutral} ({100*n_neutral/n_total:.1f}%)")
    print(f"Mean error (text_base): {df['error_text_base'].mean():.4f}")
    print(f"Mean error (final):    {df['error_final'].mean():.4f}")
    print(f"Mean error reduction:   {df['error_text_base'].mean() - df['error_final'].mean():.4f}")

    # Sign flip analysis
    sign_flipped = df[df['text_sign_correct'] & ~df['final_sign_correct']]
    sign_fixed = df[~df['text_sign_correct'] & df['final_sign_correct']]
    print(f"\n=== Sign Analysis ===")
    print(f"Sign corrected by residual: {len(sign_fixed)} ({100*len(sign_fixed)/n_total:.1f}%)")
    print(f"Sign damaged by residual:   {len(sign_flipped)} ({100*len(sign_flipped)/n_total:.1f}%)")

    # Delta analysis
    if 'delta_reg' in df.columns:
        print(f"\n=== Delta Analysis ===")
        print(f"abs(delta_reg) mean: {df['delta_reg'].abs().mean():.4f}")
        print(f"abs(delta_reg) std:  {df['delta_reg'].abs().std():.4f}")
        print(f"abs(delta_reg) p90:  {df['delta_reg'].abs().quantile(0.90):.4f}")
        print(f"abs(delta_reg) max:  {df['delta_reg'].abs().max():.4f}")

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f'P5D_residual_effect_by_group_{args.run_name}.csv')
    detail_path = os.path.join(args.output_dir, f'P5D_residual_effect_by_sample_{args.run_name}.csv')
    summary.to_csv(summary_path, index=False)
    df.to_csv(detail_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == '__main__':
    main()
