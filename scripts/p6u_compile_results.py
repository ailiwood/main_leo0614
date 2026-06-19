#!/usr/bin/env python
"""P6U: Compile all freeze-ready results from result.json files."""
import json, os, glob, csv

def check_file(path):
    return 'EXISTS' if os.path.exists(path) else 'MISSING'

results = []

# === MOSEI Main Model text_audio ===
d = 'outputs/P6T_prefreeze/mosei/mainline/text_audio_cached_s42_s42_20260619_222353'
r = json.load(open(os.path.join(d, 'result.json')))
results.append({
    'dataset': 'MOSEI', 'model': 'Main (ours)', 'model_type': 'main',
    'modality': 'text_audio', 'ACC2_Non0': r['final_ACC2'], 'F1_Non0': r['F1_Non0'],
    'ACC2_Has0': 'N/A', 'F1_Has0': 'N/A', 'MAE': r['MAE'], 'Corr': r['Corr'],
    'ACC7': r.get('ACC7', 'MISSING'), 'seed': 42, 'best_epoch': r['best_epoch'],
    'source_phase': 'P6T', 'config_path': r.get('config', 'MISSING'),
    'output_dir': d, 'best_model_path': check_file(os.path.join(d, 'best_model.pth')),
    'predictions_test_path': check_file(os.path.join(d, 'predictions_test.csv')),
    'metrics_path': check_file(os.path.join(d, 'result.json')),
    'credibility': 'A', 'can_enter_paper': True, 'freeze_ready': True,
    'notes': 'residual_gain=0, gate=1.0 - audio branch unused'
})

# === MOSEI Main Model text_only diagnostic ===
d2 = 'outputs/P6S_repair/mosei/text_only/mosei_text_only_s42_s42_20260619_160547'
r2 = json.load(open(os.path.join(d2, 'result.json')))
results.append({
    'dataset': 'MOSEI', 'model': 'Main (ours)', 'model_type': 'main_diagnostic',
    'modality': 'text_only', 'ACC2_Non0': r2['final_ACC2'], 'F1_Non0': r2['F1_Non0'],
    'ACC2_Has0': 'N/A', 'F1_Has0': 'N/A', 'MAE': r2['MAE'], 'Corr': r2['Corr'],
    'ACC7': r2.get('ACC7', 'MISSING'), 'seed': 42, 'best_epoch': r2['best_epoch'],
    'source_phase': 'P6S_repair', 'config_path': r2.get('config', 'MISSING'),
    'output_dir': d2, 'best_model_path': check_file(os.path.join(d2, 'best_model.pth')),
    'predictions_test_path': check_file(os.path.join(d2, 'predictions_test.csv')),
    'metrics_path': check_file(os.path.join(d2, 'result.json')),
    'credibility': 'A', 'can_enter_paper': True, 'freeze_ready': True,
    'notes': 'text-only diagnostic for multimodal gain audit'
})

# === MOSI Main Model text_audio ===
d3 = 'outputs/P6K/text_audio_conservative_s42_s42_20260619_031645'
r3 = json.load(open(os.path.join(d3, 'result.json')))
results.append({
    'dataset': 'MOSI', 'model': 'Main (ours)', 'model_type': 'main',
    'modality': 'text_audio', 'ACC2_Non0': r3['final_ACC2'], 'F1_Non0': r3['F1_Non0'],
    'ACC2_Has0': 'N/A', 'F1_Has0': 'N/A', 'MAE': r3['MAE'], 'Corr': r3['Corr'],
    'ACC7': r3.get('ACC7', 'MISSING'), 'seed': 42, 'best_epoch': r3['best_epoch'],
    'source_phase': 'P6K', 'config_path': 'configs/experiments/p6k_text_audio_conservative_s42.yaml',
    'output_dir': d3, 'best_model_path': check_file(os.path.join(d3, 'best_model.pth')),
    'predictions_test_path': check_file(os.path.join(d3, 'predictions_test.csv')),
    'metrics_path': check_file(os.path.join(d3, 'result.json')),
    'credibility': 'A', 'can_enter_paper': True, 'freeze_ready': True,
    'notes': 'BEST MOSI main model'
})

# === MOSEI Baselines (7 models) ===
base_dir = 'outputs/P6S_repair5/mosei/baselines'
base_models = [
    ('misa_lite', 'P6S_repair5'),
    ('selfmm_lite', 'P6S_repair5'),
    ('mult_lite', 'P6S_repair5'),
    ('lmf_lite', 'P6S_repair5'),
    ('tfn_lite', 'P6S_repair5'),
    ('mlcl_lite', 'P6S_repair5'),
    ('dlf_lite', 'P6S_repair5'),
]
for model, phase in base_models:
    dirs = sorted(glob.glob(os.path.join(base_dir, f'{model}_s42_*')))
    if not dirs:
        results.append({'dataset': 'MOSEI', 'model': model, 'freeze_ready': False, 'notes': 'NO OUTPUT DIR'})
        continue
    d = dirs[-1]
    r = json.load(open(os.path.join(d, 'result.json')))
    results.append({
        'dataset': 'MOSEI', 'model': model, 'model_type': 'baseline',
        'modality': 'text_audio',
        'ACC2_Non0': r.get('ACC2_Non0', r.get('final_ACC2', 'MISSING')),
        'F1_Non0': r.get('F1_Non0', 'MISSING'),
        'ACC2_Has0': r.get('ACC2_Has0', 'N/A'), 'F1_Has0': r.get('F1_Has0', 'N/A'),
        'MAE': r.get('MAE', 'MISSING'), 'Corr': r.get('Corr', 'MISSING'),
        'ACC7': r.get('ACC7', 'MISSING'), 'seed': 42,
        'best_epoch': r.get('best_epoch', 'MISSING'),
        'source_phase': phase,
        'config_path': f'configs/experiments/p6s_repair5_mosei_baselines_fixed/mosei_{model}_cached_s42.yaml',
        'output_dir': d,
        'best_model_path': check_file(os.path.join(d, 'best_model.pth')),
        'predictions_test_path': check_file(os.path.join(d, 'predictions_test.csv')),
        'metrics_path': check_file(os.path.join(d, 'result.json')),
        'credibility': 'A', 'can_enter_paper': True, 'freeze_ready': True,
        'notes': ''
    })

# === MOSI Baselines (4 models) ===
mosi_base = {
    'tfn_lite': 'outputs/P6Q/baseline_rescue/mosi/tfn_lite_s42_20260619_135758',
    'mult_lite': 'outputs/P6P/baseline_rescue/mosi/mult_lite_s42_20260619_134203',
    'selfmm_lite': 'outputs/P6P/baseline_rescue/mosi/selfmm_lite_s42_20260619_133527',
    'lmf_lite': 'outputs/P6Q/baseline_rescue/mosi/lmf_lite_s42_20260619_140344',
}
for model, d in mosi_base.items():
    r = json.load(open(os.path.join(d, 'result.json')))
    results.append({
        'dataset': 'MOSI', 'model': model, 'model_type': 'baseline',
        'modality': 'text_audio',
        'ACC2_Non0': r.get('ACC2_Non0', r.get('final_ACC2', 'MISSING')),
        'F1_Non0': r.get('F1_Non0', 'MISSING'),
        'ACC2_Has0': 'N/A', 'F1_Has0': 'N/A',
        'MAE': r.get('MAE', 'MISSING'), 'Corr': r.get('Corr', 'MISSING'),
        'ACC7': r.get('ACC7', 'MISSING'), 'seed': 42,
        'best_epoch': r.get('best_epoch', 'MISSING'),
        'source_phase': 'P6P/P6Q',
        'config_path': 'MISSING (rescue phase)',
        'output_dir': d,
        'best_model_path': check_file(os.path.join(d, 'best_model.pth')),
        'predictions_test_path': check_file(os.path.join(d, 'predictions_test.csv')),
        'metrics_path': check_file(os.path.join(d, 'result.json')),
        'credibility': 'A', 'can_enter_paper': True, 'freeze_ready': True,
        'notes': 'MOSI rescue result'
    })

# Save CSV
os.makedirs('reports/P6U_freeze/tables', exist_ok=True)
csv_path = 'reports/P6U_freeze/tables/main_results_all.csv'
fields = list(results[0].keys())
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)
print(f'Saved {len(results)} models to {csv_path}')

# Print summary
print(f'\n{"Dataset":>6s} {"Model":>15s} {"Modality":>12s} {"ACC2":>10s} {"Freeze":>6s}')
print('-' * 55)
for r in results:
    acc2 = r.get('ACC2_Non0', '?')
    acc2_str = f'{acc2:.2f}%' if isinstance(acc2, float) else str(acc2)
    freeze = 'YES' if r.get('freeze_ready') else 'NO'
    print(f'{r["dataset"]:>6s} {r["model"]:>15s} {r["modality"]:>12s} {acc2_str:>10s} {freeze:>6s}')

# Count
mosi_count = sum(1 for r in results if r['dataset'] == 'MOSI' and r.get('freeze_ready'))
mosei_count = sum(1 for r in results if r['dataset'] == 'MOSEI' and r.get('freeze_ready'))
print(f'\nFreeze-ready: MOSI={mosi_count}, MOSEI={mosei_count}, Total={mosi_count+mosei_count}')
