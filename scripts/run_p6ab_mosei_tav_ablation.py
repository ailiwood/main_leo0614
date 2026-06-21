"""
P6AB MOSEI TAV Ablation Runner

Sequentially runs all 7 P6AB MOSEI TAV ablation variants using the fixed 4-epoch protocol.
Uses mme_xlstm_stable conda env (PyTorch 2.11.0+cu128 for RTX 5070 Ti Blackwell support).

Run order: F0 → F1 → F2 → F3 → F4 → F5 → E1

Usage:
  /e/Anaconda3/envs/mme_xlstm_stable/python.exe scripts/run_p6ab_mosei_tav_ablation.py
  /e/Anaconda3/envs/mme_xlstm_stable/python.exe scripts/run_p6ab_mosei_tav_ablation.py --start F3
  /e/Anaconda3/envs/mme_xlstm_stable/python.exe scripts/run_p6ab_mosei_tav_ablation.py --dry-run
"""
import subprocess, sys, os, time, json, argparse
from datetime import datetime

# Fixed run order
RUN_ORDER = ['F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'E1']

CONFIG_MAP = {
    'F0': 'configs/experiments/p6ab_tav_ablation/mosei/F0_full_tav_awaf_slstm_s42.yaml',
    'F1': 'configs/experiments/p6ab_tav_ablation/mosei/F1_no_vision_text_audio_s42.yaml',
    'F2': 'configs/experiments/p6ab_tav_ablation/mosei/F2_no_audio_text_vision_s42.yaml',
    'F3': 'configs/experiments/p6ab_tav_ablation/mosei/F3_global_static_tav_s42.yaml',
    'F4': 'configs/experiments/p6ab_tav_ablation/mosei/F4_fixed_mean_tav_s42.yaml',
    'F5': 'configs/experiments/p6ab_tav_ablation/mosei/F5_awaf_no_interaction_tav_s42.yaml',
    'E1': 'configs/experiments/p6ab_tav_ablation/mosei/E1_temporal_lstm_all_s42.yaml',
}

DESC = {
    'F0': 'Full T+A+V AWAF sLSTM (control)',
    'F1': 'No vision: T+A only (vision contribution)',
    'F2': 'No audio: T+V only (audio contribution)',
    'F3': 'Global static learnable weights (sample-level AWAF contribution)',
    'F4': 'Fixed mean w_t=w_a=w_v=1/3 (dynamic fusion contribution)',
    'F5': 'AWAF without Hadamard interaction g_ta/g_tv/g_av (interaction contribution)',
    'E1': 'LSTM replaces sLSTM for audio+vision (temporal encoder contribution)',
}

PYTHON = '/e/Anaconda3/envs/mme_xlstm_stable/python.exe'
TRAIN_SCRIPT = 'scripts/train_textft_lora_mainline.py'

START_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = f'reports/P6AB_tav_causal_evidence/mosei/ablation_run_{START_TIME}.log'


def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f'[{timestamp}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def run_variant(var_id, dry_run=False):
    config_path = CONFIG_MAP[var_id]
    desc = DESC[var_id]

    log(f'{"="*70}')
    log(f'Starting {var_id}: {desc}')
    log(f'Config: {config_path}')
    log(f'{"="*70}')

    if dry_run:
        log(f'[DRY RUN] Would execute: {PYTHON} {TRAIN_SCRIPT} --config {config_path}')
        return True

    cmd = [PYTHON, TRAIN_SCRIPT, '--config', config_path]
    start_t = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd='E:/00project_code/main_leo/new_code',
            capture_output=True,
            text=True,
            timeout=14400,  # 4 hour timeout per variant
        )
        elapsed = time.time() - start_t

        # Save output
        out_log = f'reports/P6AB_tav_causal_evidence/mosei/{var_id}_train_{START_TIME}.log'
        with open(out_log, 'w') as f:
            f.write(f'STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}')

        if result.returncode == 0:
            log(f'{var_id} COMPLETED in {elapsed/60:.1f} min (exit code 0)')
            log(f'  Output log: {out_log}')
            return True
        else:
            log(f'{var_id} FAILED in {elapsed/60:.1f} min (exit code {result.returncode})')
            log(f'  STDOUT tail: {result.stdout[-500:] if result.stdout else "(empty)"}')
            log(f'  STDERR tail: {result.stderr[-500:] if result.stderr else "(empty)"}')
            log(f'  Full log: {out_log}')
            return False

    except subprocess.TimeoutExpired:
        log(f'{var_id} TIMEOUT after 4 hours')
        return False
    except Exception as e:
        log(f'{var_id} EXCEPTION: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='F0', help='Variant to start from')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--only', type=str, default='', help='Run only this variant (comma-separated)')
    args = parser.parse_args()

    log(f'P6AB MOSEI TAV Ablation Runner')
    log(f'Start time: {datetime.now().isoformat()}')
    log(f'Python: {PYTHON}')
    log(f'GPU: checking...')

    # Quick GPU check
    try:
        result = subprocess.run(
            [PYTHON, '-c', 'import torch; print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")'],
            capture_output=True, text=True, timeout=30,
        )
        log(f'  {result.stdout.strip()}')
    except Exception as e:
        log(f'  GPU check failed: {e}')

    # Determine which variants to run
    if args.only:
        to_run = [v.strip() for v in args.only.split(',')]
    else:
        try:
            start_idx = RUN_ORDER.index(args.start)
        except ValueError:
            log(f'ERROR: Unknown variant {args.start}')
            return
        to_run = RUN_ORDER[start_idx:]

    log(f'Will run: {to_run}')
    if args.dry_run:
        log('DRY RUN MODE - no training will execute')

    results = {}
    for var_id in to_run:
        success = run_variant(var_id, dry_run=args.dry_run)
        results[var_id] = 'OK' if success else 'FAILED'

        if not success and not args.dry_run:
            log(f'STOPPING: {var_id} failed. Remaining variants skipped.')
            for remaining in to_run[to_run.index(var_id)+1:]:
                results[remaining] = 'SKIPPED'
            break

    # Summary
    log(f'\n{"="*70}')
    log(f'ABLATION RUN SUMMARY')
    log(f'{"="*70}')
    for var_id, status in results.items():
        emoji = 'PASS' if status == 'OK' else ('SKIP' if status == 'SKIPPED' else 'FAIL')
        log(f'  {var_id}: {emoji} - {DESC.get(var_id, "")}')
    log(f'End time: {datetime.now().isoformat()}')

    # Save results JSON
    results_path = f'reports/P6AB_tav_causal_evidence/mosei/ablation_results_{START_TIME}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'start_time': START_TIME,
            'results': results,
            'python': PYTHON,
            'run_order': to_run,
        }, f, indent=2)
    log(f'Results saved: {results_path}')


if __name__ == '__main__':
    main()
