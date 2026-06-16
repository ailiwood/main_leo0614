"""P3B Candidate Module Selection — Run C0/C1/C2/C3 experiments."""
import sys, os, argparse, csv, time, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch; import yaml
from torch.utils.data import DataLoader
from utils.seed import set_seed
from data.dataset import TMDCMOSIDataset, collate_fn
from models.ours_xlstm_fusion import OursXLSTMFusion
from engine.trainer import Trainer

def load_cfg(path):
    with open(path) as f: return yaml.safe_load(f)

def build_ts(): return datetime.now().strftime('%Y%m%d_%H%M%S')

def run_exp(model_name, candidate, model_cfg, train_cfg, seed, device, base_out, registry):
    os.makedirs(base_out, exist_ok=True)
    ts = build_ts()
    run_id = f"{model_name}_s{seed}_{ts}"
    out_dir = os.path.join(base_out, model_name, run_id)

    # Handle C2: use Data2Vec-Audio features if enabled
    use_d2v = model_cfg.get('use_data2vec_audio', False)

    # Load data
    train_ds = TMDCMOSIDataset('train')
    if use_d2v:
        from data.dataset import TMDCMOSIDataset as DS
        # For C2, use a modified dataset that loads data2vec audio
        # Currently TMDCMOSIDataset loads all 1024d features.
        # We need to replace audio with data2vec features.
        # For now, create a wrapper that loads data2vec npy files
        import numpy as np
        class C2Dataset(DS):
            def __init__(self, split='train', d2v_dir=None):
                super().__init__(split=split)
                if d2v_dir is None:
                    d2v_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          'data', 'features_mosi_data2vec')
                self.d2v_dir = d2v_dir
                self.use_d2v = os.path.exists(d2v_dir)
            def __getitem__(self, idx):
                item = super().__getitem__(idx)
                if self.use_d2v:
                    uid = item['id']
                    d2v_path = os.path.join(self.d2v_dir, uid + '.npy')
                    if os.path.exists(d2v_path):
                        feat = np.load(d2v_path).astype(np.float32)
                        if feat.ndim == 1: feat = feat[np.newaxis, :]
                        item['audio'] = torch.from_numpy(feat).float()
                        item['audio_mask'] = torch.ones(feat.shape[0], dtype=torch.long)
                return item
        train_ds = C2Dataset('train')
        val_ds = C2Dataset('val')
        test_ds = C2Dataset('test')

    val_ds = TMDCMOSIDataset('val') if not use_d2v else C2Dataset('val')
    test_ds = TMDCMOSIDataset('test') if not use_d2v else C2Dataset('test')

    bs = train_cfg.get('batch_size', 32)
    tr_ld = DataLoader(train_ds, bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
    te_ld = DataLoader(test_ds, bs, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Build model
    set_seed(seed)
    model = OursXLSTMFusion(model_cfg)
    n = sum(p.numel() for p in model.parameters())

    lr = train_cfg.get('lr', 1e-4)
    trainer = Trainer(model, device, lr, weight_decay=train_cfg.get('weight_decay',0.01),
                       reg_loss_weight=1.0, cls_loss_weight=0.5,
                       aux_loss_weight=train_cfg.get('aux_loss_weight',0))

    epochs = train_cfg.get('epochs', 20)
    print(f"  [{model_name}] seed={seed}, params={n:,}, lr={lr}, epochs={epochs}")

    t0 = time.time()
    for ep in range(1, epochs+1):
        tl = trainer.train_epoch(tr_ld, epoch=ep)
        er = trainer.evaluate(te_ld)
        trainer.record_epoch(ep, tl, er, 0)
        if ep % 5 == 0 or ep == epochs:
            m = er['metrics']
            print(f"    E{ep:2d} | loss={tl:.3f} MAE={m['MAE']:.3f} ACC2_NZ={m['ACC2_Non0']:.1f}%")

    elapsed = time.time()-t0
    fe = trainer.evaluate(te_ld)
    trainer.save_predictions_and_weights(out_dir, fe)
    trainer.save_run(out_dir, model_cfg, f"p3b {model_name} s{seed}", epoch=epochs)

    m = fe['metrics']; a = fe['awaf_stats']
    r = {
        'model_name': model_name, 'candidate_module': candidate, 'seed': seed,
        'epochs': epochs, 'hidden_dim': model_cfg.get('hidden_dim',256),
        'lr': lr, 'batch_size': bs, 'n_params': n, 'train_time_sec': elapsed,
        'MAE': m['MAE'], 'Corr': m['Corr'], 'ACC2_Non0': m['ACC2_Non0'],
        'F1_Non0': m['F1_Non0'], 'ACC2_Has0': m['ACC2_Has0'], 'F1_Has0': m['F1_Has0'],
        'ACC7': m['ACC7'], 'w_t_mean': a['w_t_mean'], 'w_a_mean': a['w_a_mean'],
        'w_v_mean': a['w_v_mean'], 'w_t_std': a['w_t_std'], 'w_a_std': a['w_a_std'],
        'w_v_std': a['w_v_std'], 'sum_w_max_dev': a['sum_w_max_dev'],
        'confidence_level': 'C', 'paper_usable': False, 'output_dir': out_dir,
    }
    # Append registry
    if r:
        os.makedirs(os.path.dirname(registry), exist_ok=True)
        keys = list(r.keys())
        fex = os.path.exists(registry)
        with open(registry, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if not fex: w.writeheader()
            w.writerows([r])
    return r

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 2024])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--skip_c2', action='store_true')
    args = p.parse_args()

    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_out = os.path.join(BASE, 'outputs', 'P3B_candidate_module_selection', 'mosi')
    reg = os.path.join(BASE, 'reports', 'P3B_candidate_module_selection', 'P3B_candidate_results.csv')

    # C0 baseline config
    c0_cfg = {
        'text_dim':1024,'audio_dim':1024,'vision_dim':1024,'hidden_dim':256,
        'proj_dropout':0.1,'slstm_num_layers':1,'slstm_dropout':0.3,
        'slstm_bidirectional':False,'slstm_pooling':'masked_mean',
        'awaf_fusion_mode':'awaf','awaf_tau_init':1.0,'awaf_dropout':0.3,
        'awaf_modality_dropout':True,'head_reg_hidden':128,'head_cls_hidden':128,
        'head_dropout':0.3,'use_aux_heads':False,
        'use_deconv':False,'use_cme':False,'use_data2vec_audio':False,
    }
    train_cfg = {'epochs':args.epochs,'lr':1e-4,'batch_size':32,'weight_decay':0.01,'aux_loss_weight':0.0}

    # Check C2 feasibility
    c2_ok = not args.skip_c2
    if c2_ok:
        # Quick check: can we load data2vec-audio?
        try:
            from transformers import Data2VecAudioModel
            _ = Data2VecAudioModel.from_pretrained('facebook/data2vec-audio-base-960h')
            print("C2: Data2Vec-Audio model available. Will extract features during dataset loading.")
        except:
            print("C2: Data2Vec-Audio NOT available. Skipping C2.")
            c2_ok = False

    all_results = []
    for seed in args.seeds:
        # C0
        print(f"\n{'='*50}\n  C0 Baseline seed={seed}\n{'='*50}")
        r = run_exp('C0','none',c0_cfg,train_cfg,seed,args.device,base_out,reg)
        all_results.append(r)

        # C1: DEConv
        print(f"\n{'='*50}\n  C1 DEConv seed={seed}\n{'='*50}")
        c1_cfg = {**c0_cfg,'use_deconv':True}
        r = run_exp('C1','DEConv',c1_cfg,train_cfg,seed,args.device,base_out,reg)
        all_results.append(r)

        # C3: CME
        print(f"\n{'='*50}\n  C3 CME seed={seed}\n{'='*50}")
        c3_cfg = {**c0_cfg,'use_cme':True,'cme_num_heads':4,'cme_dropout':0.1}
        r = run_exp('C3','CME',c3_cfg,train_cfg,seed,args.device,base_out,reg)
        all_results.append(r)

        # C2: Data2Vec Audio (if feasible)
        if c2_ok:
            print(f"\n{'='*50}\n  C2 Data2Vec seed={seed}\n{'='*50}")
            c2_cfg = {**c0_cfg,'use_data2vec_audio':True,'audio_dim':768}
            r = run_exp('C2','Data2Vec-Audio',c2_cfg,train_cfg,seed,args.device,base_out,reg)
            all_results.append(r)

    # Summary
    print(f"\n{'='*80}\n  P3B SUMMARY ({len(all_results)} runs)\n{'='*80}")
    print(f"{'Model':<8} {'Seed':>5} {'ACC2_NZ':>8} {'MAE':>6} {'Corr':>6} {'F1_NZ':>7} {'w_t':>6} {'w_a':>6} {'w_v':>6} {'Time':>6}s")
    print('-'*85)
    for r in sorted(all_results, key=lambda x: x['ACC2_Non0'], reverse=True):
        print(f"{r['model_name']:<8} {r['seed']:>5} {r['ACC2_Non0']:>7.1f}% {r['MAE']:>6.3f} {r['Corr']:>6.3f} {r['F1_Non0']:>6.1f}% {r['w_t_mean']:>5.3f} {r['w_a_mean']:>5.3f} {r['w_v_mean']:>5.3f} {r['train_time_sec']:>5.0f}")
    print(f"\n  Registry: {reg}")

if __name__ == '__main__':
    main()
