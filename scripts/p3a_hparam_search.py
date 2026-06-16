"""
scripts/p3a_hparam_search.py — P3A C0 超参数探索
"""
import sys, os, argparse, json, time, csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from utils.seed import set_seed
from data.dataset import TMDCMOSIDataset, collate_fn
from models.ours_xlstm_fusion import OursXLSTMFusion
from engine.trainer import Trainer

def build_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def build_model_config(hd, layers, do, awaf_md, use_aux):
    return {
        'text_dim':1024,'audio_dim':1024,'vision_dim':1024,'hidden_dim':hd,
        'slstm_num_layers':layers,'slstm_dropout':do,'slstm_bidirectional':False,
        'slstm_pooling':'masked_mean','awaf_fusion_mode':'awaf','awaf_tau_init':1.0,
        'awaf_dropout':do,'awaf_modality_dropout':awaf_md,
        'head_reg_hidden':128,'head_cls_hidden':128,'head_dropout':do,
        'use_aux_heads':use_aux,'proj_dropout':0.1,
    }

def run_one(model_cfg, train_kw, output_dir, seed, device):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)
    train_ds = TMDCMOSIDataset('train'); val_ds = TMDCMOSIDataset('val'); test_ds = TMDCMOSIDataset('test')
    bs = train_kw['batch_size']
    tr_ld = DataLoader(train_ds, bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
    te_ld = DataLoader(test_ds, bs, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = OursXLSTMFusion(model_cfg); n = sum(p.numel() for p in model.parameters())
    trainer = Trainer(model, device, train_kw['lr'], weight_decay=0.01,
                       reg_loss_weight=1.0, cls_loss_weight=0.5,
                       aux_loss_weight=train_kw.get('aux_loss_weight',0))
    epochs = train_kw['epochs']; t0 = time.time()
    for ep in range(1, epochs+1):
        tl = trainer.train_epoch(tr_ld, epoch=ep)
        er = trainer.evaluate(te_ld)
        trainer.record_epoch(ep, tl, er, 0)
        if ep%5==0 or ep==epochs:
            m=er['metrics']; print(f"  E{ep:2d} | loss={tl:.3f} MAE={m['MAE']:.3f} ACC2_NZ={m['ACC2_Non0']:.1f}%")
    elapsed = time.time()-t0
    fe = trainer.evaluate(te_ld); trainer.save_predictions_and_weights(output_dir, fe)
    trainer.save_run(output_dir, model_cfg, f"p3a_s{seed}", epoch=epochs)
    m=fe['metrics']; a=fe['awaf_stats']
    return {
        'hidden_dim':model_cfg['hidden_dim'],'slstm_layers':model_cfg['slstm_num_layers'],
        'lr':train_kw['lr'],'dropout':model_cfg['head_dropout'],'batch_size':bs,
        'aux_loss_weight':train_kw.get('aux_loss_weight',0),'awaf_md':model_cfg['awaf_modality_dropout'],
        'seed':seed,'n_params':n,'total_time_sec':elapsed,
        'MAE':m['MAE'],'Corr':m['Corr'],'ACC2_Non0':m['ACC2_Non0'],'F1_Non0':m['F1_Non0'],
        'ACC2_Has0':m['ACC2_Has0'],'F1_Has0':m['F1_Has0'],'ACC7':m['ACC7'],
        'w_t_mean':a['w_t_mean'],'w_a_mean':a['w_a_mean'],'w_v_mean':a['w_v_mean'],
        'w_t_std':a['w_t_std'],'w_a_std':a['w_a_std'],'w_v_std':a['w_v_std'],
        'sum_w_max_dev':a['sum_w_max_dev'],'output_dir':output_dir,
    }

def save_registry(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results: return
    fe = list(results[0].keys()); ex = os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fe)
        if not ex: w.writeheader()
        w.writerows(results)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,nargs='+',default=[42])
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--device',type=str,default='cuda')
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): args.device='cpu'

    # Default grid (10 combos for seed=42)
    # For seed validation (--seed 2024), use --subset flag or modify directly
    grid = [
        (128,1,5e-5,0.3,32,0.0,True),   # 0: baseline 128
        (128,1,5e-5,0.1,32,0.0,True),   # 1: low dropout
        (128,2,5e-5,0.3,32,0.0,True),   # 2: 2 layers
        (256,1,5e-5,0.3,32,0.0,True),   # 3: baseline 256
        (256,1,5e-5,0.1,32,0.0,True),   # 4: low dropout
        (256,1,1e-4,0.3,32,0.0,True),   # 5: higher lr
        (256,2,5e-5,0.3,32,0.0,True),   # 6: 2 layers
        (256,1,5e-5,0.3,16,0.0,True),   # 7: smaller batch
        (256,1,5e-5,0.3,32,0.1,True),   # 8: with aux
        (256,1,5e-5,0.3,32,0.0,False),  # 9: w/o md
    ]

    base_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'outputs','P3A_c0_stability_exploration','mosi','C0')
    reg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'reports','P3A_c0_stability_exploration','P3A_c0_hparam_search.csv')

    all_r = []
    for seed in args.seed:
        for hd,layers,lr,do,bs,aux_w,awaf_md in grid:
            run_id = f"h{hd}_l{layers}_lr{lr:.0e}_do{do}_bs{bs}_aux{aux_w}_md{int(awaf_md)}_s{seed}"
            out_dir = os.path.join(base_out, f"{build_timestamp()}_{run_id}")
            print(f"\n{'='*60}\n  RUN: {run_id}\n{'='*60}")
            cfg = build_model_config(hd,layers,do,awaf_md,(aux_w>0))
            kw = {'epochs':args.epochs,'lr':lr,'batch_size':bs,'aux_loss_weight':aux_w}
            if args.dry_run: continue
            r = run_one(cfg, kw, out_dir, seed, args.device)
            r['run_id'] = run_id
            all_r.append(r)
            save_registry([r], reg_path)

    print(f"\n{'='*80}\n  P3A SUMMARY ({len(all_r)} runs)\n{'='*80}")
    print(f"{'run_id':<45} {'ACC2_NZ':>8} {'MAE':>6} {'Corr':>6} {'w_t':>6} {'w_a':>6} {'w_v':>6}")
    print("-"*83)
    for r in sorted(all_r,key=lambda x:x['ACC2_Non0'],reverse=True):
        print(f"{r['run_id']:<45} {r['ACC2_Non0']:>7.1f}% {r['MAE']:>5.3f} {r['Corr']:>5.3f} {r['w_t_mean']:>5.3f} {r['w_a_mean']:>5.3f} {r['w_v_mean']:>5.3f}")
    print(f"\n  Registry: {reg_path}")

if __name__=='__main__':
    main()
