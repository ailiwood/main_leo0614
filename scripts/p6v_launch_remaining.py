#!/usr/bin/env python
"""P6V-R: Launch remaining formal ablation experiments in order."""
import subprocess, sys, os, time, json

SCRIPT = 'scripts/train_textft_lora_mainline.py'
DEVICE = 'cuda'

# Priority 1: MOSI core (7 remaining after current awaf_no_context)
MOSI_QUEUE = [
    'awaf_no_interaction',
    'fusion_mean',
    'fusion_concat',
    'fusion_gated',
    'encoder_gru',
    'encoder_no_temporal',
]
# Priority 2: MOSEI compact (4 remaining after fusion_gated done)
MOSEI_QUEUE = [
    'fusion_mean',
    'awaf_no_interaction',
    'encoder_gru',
    'encoder_no_temporal',
]

def run_experiment(dataset, name):
    config = f'configs/experiments/p6v_ablation/{dataset}/{name}_s42.yaml'
    out_pattern = f'outputs/P6V_ablation/{dataset}/{name}_s42_'

    # Skip if already has formal result
    import glob
    existing = glob.glob(out_pattern + '*')
    for d in existing:
        rf = os.path.join(d, 'result.json')
        if os.path.exists(rf):
            r = json.load(open(rf))
            ep = r.get('epochs', r.get('epochs_run', 0))
            if ep and ep >= (4 if dataset == 'mosei' else 8):
                acc2 = r.get('final_ACC2', r.get('ACC2_Non0', 0))
                print(f'  [SKIP] {name}: already done (ep={ep}, ACC2={acc2:.2f}%)')
                return {'status': 'skip', 'reason': 'already_complete'}

    print(f'  [LAUNCH] {dataset}/{name} ...')
    start = time.time()
    cmd = ['python', SCRIPT, '--config', config, '--device', DEVICE]

    # Write command.txt
    for d in existing:
        with open(os.path.join(d, 'command.txt'), 'w') as f:
            f.write(' '.join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - start

    # Parse result
    has_nan = 'nan' in (result.stdout + result.stderr).lower()
    success = result.returncode == 0 and not has_nan

    # Find output dir
    new_dirs = sorted(glob.glob(out_pattern + '*'))
    out_dir = new_dirs[-1] if new_dirs else 'unknown'

    return {
        'status': 'pass' if success else 'fail',
        'name': name, 'dataset': dataset,
        'elapsed_min': round(elapsed/60, 1),
        'output': out_dir,
    }

if __name__ == '__main__':
    print('P6V-R: Remaining Ablation Launcher')
    print(f'MOSI queue: {len(MOSI_QUEUE)} experiments')
    print(f'MOSEI queue: {len(MOSEI_QUEUE)} experiments')
    print(f'Total: {len(MOSI_QUEUE) + len(MOSEI_QUEUE)} experiments')
    print()

    results = []

    # MOSI first (faster, each ~5 min)
    for name in MOSI_QUEUE:
        r = run_experiment('mosi', name)
        results.append(r)
        status = r['status']
        print(f'    -> {status.upper()} ({r.get(\"elapsed_min\",0)} min)')

    # Then MOSEI (slower, each ~30 min)
    for name in MOSEI_QUEUE:
        r = run_experiment('mosei', name)
        results.append(r)
        status = r['status']
        print(f'    -> {status.upper()} ({r.get(\"elapsed_min\",0)} min)')

    passed = sum(1 for r in results if r['status'] == 'pass')
    skipped = sum(1 for r in results if r['status'] == 'skip')
    failed = sum(1 for r in results if r['status'] == 'fail')
    print(f'\n=== DONE: {passed} pass, {skipped} skip, {failed} fail ===')
