"""
engine/strict_trainer.py — Strict Protocol Trainer

Key changes from P4A:
  1. train ONLY on train split
  2. val for early_stopping / best_checkpoint selection
  3. test ONLY ONCE at the end, on the best checkpoint
  4. metrics include both cls_logit and reg_sign ACC2
"""
import os, json, time, csv
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
from utils.metrics import compute_all_metrics
from models.fusion.awaf import save_awaf_weights_csv


class StrictTrainer:
    def __init__(self, model, device='cuda', lr=5e-5, weight_decay=0.01,
                 reg_loss_weight=1.0, cls_loss_weight=0.5, aux_loss_weight=0.0,
                 awaf_entropy_reg_weight=0.0, sign_consistency_weight=0.0,
                 delta_reg_weight=0.0, use_amp=False, eps=1e-8):
        self.model = model.to(device)
        self.device = device; self.lr = lr; self.weight_decay = weight_decay
        self.reg_loss_weight = reg_loss_weight; self.cls_loss_weight = cls_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.awaf_entropy_reg_weight = awaf_entropy_reg_weight
        self.sign_consistency_weight = sign_consistency_weight
        self.delta_reg_weight = delta_reg_weight
        self.use_amp = use_amp; self.eps = eps
        self.l1 = nn.L1Loss(); self.bce = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=8)
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        self.val_history: List[Dict] = []
        self.best_val_mae = float('inf'); self.best_epoch = 0
        self.best_state = None
        self.current_lr = lr

    def _compute_loss(self, output, labels):
        labels = labels.view(-1,1).float().to(self.device)
        p = (labels>=0).float()
        rl = self.l1(output['reg'], labels)
        cl = self.bce(output['cls'], p) if self.cls_loss_weight>0 else torch.tensor(0.,device=self.device)
        al = torch.tensor(0.,device=self.device)
        if output.get('aux') and self.aux_loss_weight>0:
            n=0
            for k,v in output['aux'].items():
                if k.endswith('_reg'): al+=self.l1(v,labels); n+=1
                elif k.endswith('_cls'): al+=self.bce(v,p); n+=1
            if n>0: al/=n
        er = torch.tensor(0.,device=self.device)
        w = output.get('awaf_weights')
        if w is not None and self.awaf_entropy_reg_weight>0:
            wc = w.clamp(self.eps)
            ent = -(wc*torch.log(wc)).sum(-1).mean()
            er = -self.awaf_entropy_reg_weight * ent
        # Sign consistency: encourage reg_pred sign to match label sign
        sc = torch.tensor(0.,device=self.device)
        if self.sign_consistency_weight > 0:
            sign_target = (labels >= 0).float().view(-1)
            sign_logit = 2.0 * output['reg'].view(-1)  # scale factor for sigmoid steepness
            sc = self.bce(sign_logit, sign_target)
        # Delta regularization (P5C): encourage small residual correction
        dr = torch.tensor(0.,device=self.device)
        delta_reg = output.get('delta_reg')
        if delta_reg is not None and self.delta_reg_weight > 0:
            dr = delta_reg.abs().mean()
        total = self.reg_loss_weight*rl + self.cls_loss_weight*cl + self.aux_loss_weight*al + er + self.sign_consistency_weight*sc + self.delta_reg_weight*dr
        return {'total':total,'reg':rl,'cls':cl,'aux':al,'entropy_reg':er,'sign_consistency':sc,'delta_reg':dr}

    def _to_device(self, x):
        if isinstance(x, torch.Tensor): return x.to(self.device)
        return x  # list or other non-tensor type

    def train_epoch(self, loader, epoch=1):
        self.model.train(); total=0.0; cnt=0
        for batch in tqdm(loader, desc=f'Train E{epoch}', leave=False):
            txt=self._to_device(batch['text']); aud=batch['audio'].to(self.device)
            vis=batch['vision'].to(self.device); lbl=batch['label'].to(self.device)
            tm=batch.get('text_mask'); am=batch.get('audio_mask'); vm=batch.get('vision_mask')
            if isinstance(tm, torch.Tensor): tm=tm.to(self.device)
            if isinstance(am, torch.Tensor): am=am.to(self.device)
            if isinstance(vm, torch.Tensor): vm=vm.to(self.device)
            self.opt.zero_grad()
            out = self.model(txt,aud,vis,text_mask=tm,audio_mask=am,vision_mask=vm)
            if torch.isnan(out['reg']).any(): continue
            losses = self._compute_loss(out,lbl)
            loss = losses['total']
            if torch.isnan(loss): continue
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.opt.step()
            bs = aud.size(0)
            total+=loss.item()*bs; cnt+=bs
        return total/cnt if cnt>0 else float('nan')

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval(); all_reg,all_cls,all_lbl,all_w=[],[],[],[]
        all_reg_base,all_delta_reg,all_cls_base,all_delta_cls=[],[],[],[]
        all_gate_reg,all_gate_cls=[],[]
        total_loss=0.0; cnt=0; all_ids=[]
        for batch in tqdm(loader,desc='Eval',leave=False):
            txt=self._to_device(batch['text']); aud=batch['audio'].to(self.device)
            vis=batch['vision'].to(self.device); lbl=batch['label'].to(self.device)
            tm=batch.get('text_mask'); am=batch.get('audio_mask'); vm=batch.get('vision_mask')
            if isinstance(tm, torch.Tensor): tm=tm.to(self.device)
            if isinstance(am, torch.Tensor): am=am.to(self.device)
            if isinstance(vm, torch.Tensor): vm=vm.to(self.device)
            out = self.model(txt,aud,vis,text_mask=tm,audio_mask=am,vision_mask=vm)
            losses = self._compute_loss(out,lbl)
            bs_eval = aud.size(0)
            total_loss+=losses['total'].item()*bs_eval; cnt+=bs_eval
            all_reg.append(out['reg'].cpu()); all_cls.append(out['cls'].cpu())
            all_lbl.append(lbl.cpu()); all_w.append(out['awaf_weights'].cpu())
            # P5D: collect text_base and delta for residual analysis
            if out.get('reg_text_base') is not None:
                all_reg_base.append(out['reg_text_base'].cpu())
                all_delta_reg.append(out['delta_reg'].cpu())
            if out.get('cls_text_base') is not None:
                all_cls_base.append(out['cls_text_base'].cpu())
                all_delta_cls.append(out['delta_cls'].cpu())
            # P5D: collect residual gates if present
            if out.get('residual_gate_reg') is not None:
                all_gate_reg.append(out['residual_gate_reg'].cpu())
            if out.get('residual_gate_cls') is not None:
                all_gate_cls.append(out['residual_gate_cls'].cpu())
            ids = batch.get('id',[str(i) for i in range(len(lbl))]); all_ids.extend(ids)
        rp=torch.cat(all_reg); cp=torch.cat(all_cls); tg=torch.cat(all_lbl)
        aw=torch.cat(all_w); avg_l=total_loss/cnt if cnt>0 else float('nan')
        # Both cls and regsign metrics
        m_cls = compute_all_metrics(rp,cp,tg)
        rs = torch.where(rp>=0,1.0,-1.0)
        m_reg = compute_all_metrics(rp,rs,tg)
        wc=aw.clamp(1e-8); ent=-(wc*torch.log(wc)).sum(-1)
        aw_stats={
            'w_t_mean':float(aw[:,0].mean()),'w_a_mean':float(aw[:,1].mean()),'w_v_mean':float(aw[:,2].mean()),
            'w_t_std':float(aw[:,0].std()),'w_a_std':float(aw[:,1].std()),'w_v_std':float(aw[:,2].std()),
            'awaf_entropy_mean':float(ent.mean()),'awaf_entropy_std':float(ent.std()),
            'sum_w_max_dev':float((aw.sum(-1)-1).abs().max()),
            'collapse_ratio_0.8':float((aw[:,0]>0.8).float().mean()),
            'collapse_ratio_0.9':float((aw[:,0]>0.9).float().mean()),
        }
        result = {'reg_preds':rp,'cls_preds':cp,'targets':tg,'awaf_weights':aw,
                'metrics_cls':m_cls,'metrics_regsign':m_reg,'awaf_stats':aw_stats,
                'ids':all_ids,'loss':avg_l}
        # P5D: include text_base and delta for residual analysis
        if all_reg_base:
            result['reg_text_base'] = torch.cat(all_reg_base)
            result['delta_reg'] = torch.cat(all_delta_reg)
        if all_cls_base:
            result['cls_text_base'] = torch.cat(all_cls_base)
            result['delta_cls'] = torch.cat(all_delta_cls)
        if all_gate_reg:
            result['residual_gate_reg'] = torch.cat(all_gate_reg)
        if all_gate_cls:
            result['residual_gate_cls'] = torch.cat(all_gate_cls)
        return result

    def check_val(self, epoch, val_loader):
        """Evaluate on val ONLY. test_loader must NOT be used here.
        Updates best checkpoint based on val MAE."""
        val_r = self.evaluate(val_loader)
        val_mae = val_r['metrics_regsign']['MAE']
        record = {'epoch':epoch,'val_loss':val_r['loss'],
                  'val_MAE':val_r['metrics_regsign']['MAE'],
                  'val_ACC2_Non0_regsign':val_r['metrics_regsign']['ACC2_Non0'],
                  'val_ACC2_Non0_cls':val_r['metrics_cls']['ACC2_Non0']}
        self.val_history.append(record)
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.best_epoch = epoch
            self.best_state = {k:v.detach().clone().cpu() for k,v in self.model.state_dict().items()}
        self.scheduler.step(val_mae)
        self.current_lr = self.opt.param_groups[0]['lr']
        return val_r, record

    def final_test(self, test_loader):
        """Evaluate on test using best checkpoint (selected by val). Called ONCE after training."""
        if self.best_state is not None:
            current = {k:v.detach().clone().cpu() for k,v in self.model.state_dict().items()}
            self.model.load_state_dict(self.best_state)
        test_r = self.evaluate(test_loader)
        if self.best_state is not None:
            self.model.load_state_dict(current)
        return test_r

    def save_val_csv(self, path):
        if not self.val_history: return
        keys = list(self.val_history[0].keys())
        with open(path,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(self.val_history)

    def plot_val_curves(self, out_dir):
        if not self.val_history: return
        eps = [r['epoch'] for r in self.val_history]
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot(eps,[r['val_MAE'] for r in self.val_history],'r-',label='Val MAE')
        ax.plot(eps,[r['val_ACC2_Non0_regsign'] for r in self.val_history],'g-',label='Val ACC2_NZ_regsign')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True,alpha=0.3)
        fig.savefig(os.path.join(out_dir,'val_curves.png'),dpi=150); plt.close(fig)

    def save_run(self, out_dir, config, cmd, epoch, test_result, save_last_pth=False):
        os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,'config.json'),'w') as f: json.dump(config,f,indent=2)
        with open(os.path.join(out_dir,'command.txt'),'w') as f: f.write(cmd)
        self.save_val_csv(os.path.join(out_dir,'val_metrics_epoch.csv'))
        # Test final
        m=test_result; mr=m['metrics_regsign']; mc=m['metrics_cls']; a=m['awaf_stats']
        test_metrics = {
            'best_epoch_by_val':self.best_epoch,'best_val_MAE':self.best_val_mae,
            'ACC2_Non0_regsign':mr['ACC2_Non0'],'F1_Non0_regsign':mr['F1_Non0'],
            'ACC2_Has0_regsign':mr['ACC2_Has0'],'F1_Has0_regsign':mr['F1_Has0'],
            'ACC2_Non0_cls':mc['ACC2_Non0'],'F1_Non0_cls':mc['F1_Non0'],
            'MAE':mr['MAE'],'Corr':mr['Corr'],'ACC7':mr['ACC7'],
            **a
        }
        with open(os.path.join(out_dir,'test_metrics_final.json'),'w') as f: json.dump(test_metrics,f,indent=2)
        # Predictions
        with open(os.path.join(out_dir,'predictions_test.csv'),'w',newline='') as f:
            w=csv.writer(f); w.writerow(['sample_id','reg_pred','cls_logit','target'])
            rp=m['reg_preds'].numpy().flatten(); cp=m['cls_preds'].numpy().flatten()
            tg=m['targets'].numpy().flatten(); ids=m.get('ids',range(len(rp)))
            for i in range(len(rp)): w.writerow([ids[i] if i<len(ids) else i,rp[i],cp[i],tg[i]])
        save_awaf_weights_csv(m['awaf_weights'],m.get('ids',[]),os.path.join(out_dir,'awaf_weights_test.csv'))
        # P5D: text_base_delta_test.csv for residual analysis
        fieldnames = ['sample_id','label','reg_text_base','delta_reg','delta_scale_reg','reg_final',
                      'cls_text_base','delta_cls','delta_scale_cls','cls_final',
                      'awaf_w_t','awaf_w_a','awaf_w_v']
        if m.get('residual_gate_reg') is not None:
            fieldnames.extend(['residual_gate_reg','residual_gate_cls'])
        with open(os.path.join(out_dir,'text_base_delta_test.csv'),'w',newline='') as f:
            w=csv.writer(f); w.writerow(fieldnames)
            for i in range(len(rp)):
                row = [ids[i] if i<len(ids) else i, tg[i],
                       m.get('reg_text_base',rp)[i].item() if m.get('reg_text_base') is not None else '',
                       m.get('delta_reg',rp)[i].item() if m.get('delta_reg') is not None else '',
                       '', rp[i],
                       m.get('cls_text_base',cp)[i].item() if m.get('cls_text_base') is not None else '',
                       m.get('delta_cls',cp)[i].item() if m.get('delta_cls') is not None else '',
                       '', cp[i],
                       m['awaf_weights'][i,0].item(), m['awaf_weights'][i,1].item(), m['awaf_weights'][i,2].item()]
                if m.get('residual_gate_reg') is not None:
                    row.append(m['residual_gate_reg'][i].item())
                    row.append(m['residual_gate_cls'][i].item())
                w.writerow(row)
        torch.save({'epoch':epoch,'model_state_dict':self.best_state or self.model.state_dict(),
                    'best_val_mae':self.best_val_mae,'best_epoch':self.best_epoch,
                    'test_metrics':test_metrics},os.path.join(out_dir,'best_model.pth'))
        if save_last_pth:
            torch.save({'epoch':epoch,'model_state_dict':self.model.state_dict()},os.path.join(out_dir,'last_model.pth'))
        self.plot_val_curves(out_dir)
