#!/usr/bin/env python
"""P6U: Status checker — verifies all freeze-ready models and their artifacts."""
import sys, os, json, glob

# Define expected models
EXPECTED = {
    'MOSEI': {
        'main': ['outputs/P6T_prefreeze/mosei/mainline/text_audio_cached_s42_s42_20260619_222353'],
        'diagnostic': ['outputs/P6S_repair/mosei/text_only/mosei_text_only_s42_s42_20260619_160547'],
        'baselines': sorted(glob.glob('outputs/P6S_repair5/mosei/baselines/*_s42_*')),
    },
    'MOSI': {
        'main': ['outputs/P6K/text_audio_conservative_s42_s42_20260619_031645'],
        'baselines': [
            'outputs/P6P/baseline_rescue/mosi/mult_lite_s42_20260619_134203',
            'outputs/P6P/baseline_rescue/mosi/selfmm_lite_s42_20260619_133527',
            'outputs/P6Q/baseline_rescue/mosi/tfn_lite_s42_20260619_135758',
            'outputs/P6Q/baseline_rescue/mosi/lmf_lite_s42_20260619_140344',
        ],
    }
}

ARTIFACTS = ['result.json', 'best_model.pth', 'predictions_test.csv', 'config.yaml']

def check_artifacts(d):
    return {a: os.path.exists(os.path.join(d, a)) for a in ARTIFACTS}

print('P6U STATUS CHECK')
print('=' * 60)

total_ok = 0
total_issues = 0

for dataset, categories in EXPECTED.items():
    print(f'\n--- {dataset} ---')
    for cat, dirs in categories.items():
        for d in dirs:
            if not os.path.isdir(d):
                print(f'  [MISSING DIR] {d}')
                total_issues += 1
                continue
            arts = check_artifacts(d)
            missing = [k for k, v in arts.items() if not v]
            if missing:
                print(f'  [ISSUE] {os.path.basename(d)}: missing {missing}')
                total_issues += 1
            else:
                # Read ACC2
                try:
                    r = json.load(open(os.path.join(d, 'result.json')))
                    acc2 = r.get('ACC2_Non0', r.get('final_ACC2', '?'))
                    acc2_str = f'{acc2:.2f}%' if isinstance(acc2, float) else str(acc2)
                except:
                    acc2_str = '?'
                print(f'  [OK] {os.path.basename(d):>50s} ACC2={acc2_str}')
                total_ok += 1

# Check freeze configs
print(f'\n--- FREEZE CONFIGS ---')
config_dir = 'configs/experiments/p6u_freeze'
configs = sorted(glob.glob(f'{config_dir}/**/*.yaml', recursive=True))
for c in configs:
    print(f'  [OK] {c}')
print(f'  Total configs: {len(configs)}')

# Check code imports
print(f'\n--- CODE IMPORT TESTS ---')
import_tests = {
    'metrics': 'from utils.metrics import compute_all_metrics',
    'data': 'from data.textft_multimodal_dataset import collate_textft',
    'baseline_base': 'from models.baselines.base_baseline import BaseBaseline',
    'mult_lite': 'from models.baselines.mult_lite import MulTLite',
}
for name, imp in import_tests.items():
    try:
        exec(imp)
        print(f'  [OK] {name}')
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        total_issues += 1

print(f'\n=== STATUS: {total_ok} OK, {total_issues} issues ===')
