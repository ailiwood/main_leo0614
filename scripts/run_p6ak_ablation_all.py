"""
P6AK-G2: Automated Sequential Ablation Runner
Runs all 5 variants × 3 seeds = 15 training runs sequentially.
Uses P6K checkpoint initialization for all.
"""
import subprocess, sys, os, time, json, csv
from datetime import datetime

PYTHON = r'E:\Anaconda3\envs\mme_xlstm_stable\python.exe'
TRAIN_SCRIPT = 'scripts/train_textft_lora_mainline.py'
P6K_CKPT = 'outputs/P6K/text_audio_conservative_s42_s42_20260619_031645/best_model.pth'
CONFIG_DIR = 'configs/experiments/p6ak_mosi_final_ablation'
OUTPUT_ROOT = 'outputs/P6AK_mosi_final_ablation'
LOG_FILE = 'reports/P6AK_mosi_final_ablation/G2_ablation_run.log'

VARIANTS = [
    'A1_text_only_p6k_init',
    'A2_no_audio_correction',
    'A3_no_vision_correction',
    'A4_no_reliability_gate',
    'A5_no_interaction',
]
SEEDS = [42, 2024, 3407]

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def run_one(variant, seed):
    config_path = os.path.join(CONFIG_DIR, f'{variant}_s{seed}.yaml')
    log(f'Starting {variant} seed={seed}')

    start_t = time.time()
    try:
        result = subprocess.run(
            [PYTHON, TRAIN_SCRIPT, '--config', config_path, '--init_checkpoint', P6K_CKPT],
            cwd=r'E:\00project_code\main_leo\new_code',
            capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.time() - start_t

        # Save output
        out_log = f'reports/P6AK_mosi_final_ablation/logs/{variant}_s{seed}.log'
        os.makedirs(os.path.dirname(out_log), exist_ok=True)
        with open(out_log, 'w') as f:
            f.write(f'STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}')

        if result.returncode == 0:
            # Find result
            for line in result.stdout.split('\n'):
                if 'final_ACC2' in line or 'Text-base ACC2' in line:
                    log(f'  {line.strip()}')
            log(f'  {variant} s{seed}: OK ({elapsed/60:.1f}min)')
            return True
        else:
            log(f'  {variant} s{seed}: FAILED ({elapsed/60:.1f}min, code={result.returncode})')
            log(f'  STDERR: {result.stderr[-300:]}')
            return False
    except subprocess.TimeoutExpired:
        log(f'  {variant} s{seed}: TIMEOUT')
        return False
    except Exception as e:
        log(f'  {variant} s{seed}: ERROR: {e}')
        return False

def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log(f'P6AK-G2 Ablation Runner Starting')
    log(f'Variants: {VARIANTS}')
    log(f'Seeds: {SEEDS}')
    log(f'Total runs: {len(VARIANTS) * len(SEEDS)}')

    results = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            key = f'{variant}_s{seed}'
            success = run_one(variant, seed)
            results[key] = 'OK' if success else 'FAIL'

    # Summary
    log(f'\n{"="*60}')
    log(f'ABLATION COMPLETE')
    log(f'{"="*60}')
    ok_count = sum(1 for v in results.values() if v == 'OK')
    log(f'Passed: {ok_count}/{len(results)}')
    for k, v in results.items():
        log(f'  {k}: {v}')

    # Save results
    with open('reports/P6AK_mosi_final_ablation/G2_run_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log(f'Results saved.')

if __name__ == '__main__':
    main()
