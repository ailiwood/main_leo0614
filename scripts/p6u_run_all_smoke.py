#!/usr/bin/env python
"""P6U: Smoke test runner — 1 epoch, 5 batches per model. Validates all freeze configs can launch."""
import sys, os, subprocess, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMOKE_MATRIX = [
    # MOSEI baselines (fast ~2 min each)
    ('mosei', 'baseline', 'mosei/baselines/misa_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/selfmm_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/mult_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/lmf_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/tfn_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/mlcl_lite_s42.yaml'),
    ('mosei', 'baseline', 'mosei/baselines/dlf_lite_s42.yaml'),
    # MOSI baselines
    ('mosi', 'baseline', 'mosi/baselines/tfn_lite_s42.yaml'),
    ('mosi', 'baseline', 'mosi/baselines/mult_lite_s42.yaml'),
    ('mosi', 'baseline', 'mosi/baselines/selfmm_lite_s42.yaml'),
    ('mosi', 'baseline', 'mosi/baselines/lmf_lite_s42.yaml'),
    # Main models (slower ~15 min each for 1 epoch)
    ('mosei', 'main', 'mosei/main_text_audio_s42.yaml'),
    ('mosi', 'main', 'mosi/main_text_audio_s42.yaml'),
]

CONFIG_BASE = 'configs/experiments/p6u_freeze'
OUTPUT_BASE = 'outputs/P6U_smoke'

def run_smoke(dataset, model_type, config_rel):
    config_path = os.path.join(CONFIG_BASE, config_rel)
    if not os.path.exists(config_path):
        return {'status': 'SKIP', 'reason': f'Config missing: {config_path}'}

    model_name = config_rel.split('/')[-1].replace('.yaml', '')

    if model_type == 'baseline':
        script = 'scripts/train_baseline_lite.py'
        cmd = ['python', script, '--config', config_path, '--device', 'cuda',
               '--max_epochs', '1', '--limit_batches', '5']
    else:
        script = 'scripts/train_textft_lora_mainline.py'
        cmd = ['python', script, '--config', config_path, '--device', 'cuda',
               '--smoke']

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())
        elapsed = time.time() - start

        # Check for NaN or collapse in output
        stdout = result.stdout + result.stderr
        has_nan = 'nan' in stdout.lower() or 'NaN' in stdout
        has_collapse = 'COLLAPSE' in stdout
        success = result.returncode == 0 and not has_nan and not has_collapse

        return {
            'status': 'PASS' if success else 'FAIL',
            'model': model_name, 'dataset': dataset,
            'exit_code': result.returncode,
            'elapsed_s': round(elapsed, 1),
            'has_nan': has_nan,
            'has_collapse': has_collapse,
        }
    except subprocess.TimeoutExpired:
        return {'status': 'TIMEOUT', 'model': model_name, 'dataset': dataset}
    except Exception as e:
        return {'status': 'ERROR', 'model': model_name, 'dataset': dataset, 'reason': str(e)}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    print(f'P6U SMOKE RUNNER — {len(SMOKE_MATRIX)} models')
    print(f'Config base: {CONFIG_BASE}')
    print(f'Output base: {OUTPUT_BASE}')
    print(f'Device: {args.device}')
    print()

    results = []
    for i, (dataset, mtype, config) in enumerate(SMOKE_MATRIX):
        print(f'[{i+1}/{len(SMOKE_MATRIX)}] {config} ...', end=' ', flush=True)
        r = run_smoke(dataset, mtype, config)
        results.append(r)
        status = r['status']
        if status == 'PASS':
            print(f'PASS ({r.get("elapsed_s",0)}s)')
        elif status == 'FAIL':
            print(f'FAIL (exit={r.get("exit_code","?")}, nan={r.get("has_nan",False)})')
        else:
            print(f'{status}: {r.get("reason","?")}')

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] != 'PASS')

    print(f'\n=== SMOKE RESULTS: {passed}/{len(results)} PASS, {failed} FAIL ===')

    # Save results
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(os.path.join(OUTPUT_BASE, 'smoke_results.json'), 'w') as f:
        json.dump({'results': results, 'passed': passed, 'failed': failed,
                    'timestamp': time.strftime('%Y%m%d_%H%M%S')}, f, indent=2)
    print(f'Saved: {OUTPUT_BASE}/smoke_results.json')
