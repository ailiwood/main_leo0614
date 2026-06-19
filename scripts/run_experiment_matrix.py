#!/usr/bin/env python
"""P6R: Unified experiment matrix runner - dry_run/status/run_one."""
import sys, os, json, csv, argparse, subprocess, yaml

REGISTRY = 'reports/experiment_registry.csv'


def load_registry():
    if not os.path.exists(REGISTRY):
        return []
    with open(REGISTRY, 'r') as f:
        return list(csv.DictReader(f))


def load_matrix(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_dry_run(matrix):
    """Print all planned experiments."""
    for ds_name, ds_cfg in matrix.get('datasets', {}).items():
        for model in matrix.get('models', []):
            for seed in matrix.get('seeds', [42]):
                name = model.get('name', 'unknown')
                mode = model.get('modality', 'text_audio')
                config = model.get(f'{ds_name}_config', model.get('config', ''))
                if not config:
                    continue
                script = model.get('script', 'scripts/train_baseline_lite.py')
                cmd = f'python {script} --config {config} --device cuda'
                print(f'[{ds_name}] {name} s{seed} ({mode}): {cmd}')


def cmd_status(matrix):
    """Check status of all experiments via registry + output dirs."""
    reg = load_registry()
    print(f'Registry: {len(reg)} entries')
    phases = set(r.get('phase', '') for r in reg)
    for p in sorted(phases):
        count = sum(1 for r in reg if r.get('phase') == p)
        completed = sum(1 for r in reg if r.get('phase') == p and r.get('status', '') == 'full_train_success')
        print(f'  {p}: {count} runs, {completed} completed')

    # Check output dirs
    for root in ['outputs/P6R', 'outputs/P6Q', 'outputs/P6P', 'outputs/P6K']:
        if os.path.isdir(root):
            count = 0
            for dirpath, dirnames, filenames in os.walk(root):
                for f in filenames:
                    if f == 'result.json':
                        count += 1
            print(f'  {root}: {count} result.json files found')


def cmd_run_one(spec, matrix, execute=False):
    """Run a single experiment: dataset:model:modality:seed"""
    parts = spec.split(':')
    if len(parts) < 3:
        print(f'Usage: --run_one dataset:model:modality[:seed]')
        return
    ds, model_name, mode = parts[0], parts[1], parts[2]
    seed = int(parts[3]) if len(parts) > 3 else 42

    # Find model config in matrix
    for m in matrix.get('models', []):
        if m.get('name') == model_name:
            config = m.get(f'{ds}_config', m.get('config', ''))
            if not config:
                print(f'No config for {ds}:{model_name}')
                return
            script = m.get('script', 'scripts/train_baseline_lite.py')
            cmd = f'python {script} --config {config} --device cuda'
            print(f'RUN: {cmd}')
            if execute:
                subprocess.run(cmd, shell=True)
            return
    print(f'Model {model_name} not found in matrix')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', default='configs/experiment_matrices/p6r_matrix_draft.yaml')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--run_one', type=str, default='')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--resume_failed', action='store_true')
    args = p.parse_args()

    matrix = load_matrix(args.matrix) if os.path.exists(args.matrix) else {'models': []}

    if args.dry_run:
        cmd_dry_run(matrix)
    elif args.status:
        cmd_status(matrix)
    elif args.run_one:
        cmd_run_one(args.run_one, matrix, args.execute)
    elif args.resume_failed:
        print('Resume failed: listing failed entries only (not executing)')
        reg = load_registry()
        failed = [r for r in reg if r.get('status') in ('failed', 'interrupted')]
        for r in failed:
            print(f"  {r.get('phase')} {r.get('model')} {r.get('dataset')} s{r.get('seed')}: {r.get('notes', '')}")
        print(f'Total failed: {len(failed)}. Use --execute to retry.')
    else:
        print('Usage: --dry_run | --status | --run_one X | --resume_failed')


if __name__ == '__main__':
    main()
