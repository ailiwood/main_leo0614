#!/usr/bin/env python
"""P6G: TextFT-LoRA-AWAF MOSI training with PEFT."""
import sys,os,json,time,torch,numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.textft_multimodal_dataset import TextFTMultimodalDataset, collate_textft
from models.encoders.slstm import SLSTMEncoder
from models.pooling.attention_pooling import MaskedAttentionPooling
from models.fusion.awaf import AdaptiveWeightedAttentionFusion
from models.modules.uncertainty_residual_gate import UncertaintyGuidedResidualGate
from utils.metrics import compute_all_metrics

DEVICE='cuda';BATCH=4;ACCUM=4;EPOCHS=30;LR=5e-5;SEED=42;H=256
torch.manual_seed(SEED)

# Data
train_ds=TextFTMultimodalDataset(split='train',formal_mode=True)
val_ds=TextFTMultimodalDataset(split='val',formal_mode=True)
test_ds=TextFTMultimodalDataset(split='test',formal_mode=True)
print(f'Data: Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}')
tl=DataLoader(train_ds,BATCH,shuffle=True,collate_fn=collate_textft)
vl=DataLoader(val_ds,BATCH,shuffle=False,collate_fn=collate_textft)
tl_test=DataLoader(test_ds,BATCH,shuffle=False,collate_fn=collate_textft)

# RoBERTa + LoRA
print('Loading RoBERTa-LoRA...')
base=AutoModel.from_pretrained('roberta-large')
lora_cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['query','key','value','dense'])
roberta=get_peft_model(base,lora_cfg)
tokenizer=AutoTokenizer.from_pretrained('roberta-large')
D=roberta.config.hidden_size

# Text MLP head
txt_mlp=torch.nn.Sequential(torch.nn.Linear(D,512),torch.nn.LayerNorm(512),torch.nn.GELU(),torch.nn.Dropout(0.1),torch.nn.Linear(512,H),torch.nn.LayerNorm(H),torch.nn.GELU(),torch.nn.Dropout(0.1)).to(DEVICE)
reg_th=torch.nn.Linear(H,1).to(DEVICE);cls_th=torch.nn.Linear(H,1).to(DEVICE)

# Audio branch
ap=torch.nn.Sequential(torch.nn.Linear(768,H),torch.nn.LayerNorm(H),torch.nn.GELU(),torch.nn.Dropout(0.2)).to(DEVICE)
aslstm=SLSTMEncoder(H,H,1,dropout=0.2,bidirectional=False,pooling='masked_mean').to(DEVICE)
apool=MaskedAttentionPooling(H,dropout=0.2).to(DEVICE)

# Vision branch
vp=torch.nn.Sequential(torch.nn.Linear(768,H),torch.nn.LayerNorm(H),torch.nn.GELU(),torch.nn.Dropout(0.2)).to(DEVICE)
vslstm=SLSTMEncoder(H,H,1,dropout=0.2,bidirectional=False,pooling='masked_mean').to(DEVICE)
vpool=MaskedAttentionPooling(H,dropout=0.2).to(DEVICE)

# AWAF + Gate
awaf=AdaptiveWeightedAttentionFusion(H,fusion_mode='awaf',tau_init=1.0,dropout=0.1,use_modality_dropout=True).to(DEVICE)
gate=UncertaintyGuidedResidualGate(H,gate_hidden_dim=128,dropout=0.1).to(DEVICE)

# Delta experts
def md(): return torch.nn.Sequential(torch.nn.Linear(H,H//2),torch.nn.LayerNorm(H//2),torch.nn.GELU(),torch.nn.Dropout(0.2),torch.nn.Linear(H//2,1)).to(DEVICE)
drt,dra,drv=md(),md(),md();dct,dca,dcv=md(),md(),md()
dsr=torch.nn.Parameter(torch.tensor(0.05,device=DEVICE))
dsc=torch.nn.Parameter(torch.tensor(0.05,device=DEVICE))

# Collect all params
params=list(roberta.parameters())+list(txt_mlp.parameters())+list(reg_th.parameters())+list(cls_th.parameters())
for m in [ap,aslstm,apool,vp,vslstm,vpool,awaf,gate,drt,dra,drv,dct,dca,dcv]: params+=list(m.parameters())
params+=[dsr,dsc]
trainable=sum(p.numel() for p in params if p.requires_grad)
print(f'Trainable: {trainable/1e6:.1f}M')

opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.01)
scaler=torch.amp.GradScaler('cuda')
best_mae=float('inf');best_state=None

def forward_batch(batch):
    ids,am=batch['input_ids'].to(DEVICE),batch['attention_mask'].to(DEVICE)
    a=batch['audio'].to(DEVICE);am_a=batch['audio_mask'].to(DEVICE)
    v=batch['vision'].to(DEVICE);vm_v=batch['vision_mask'].to(DEVICE)
    lbl=batch['label'].to(DEVICE)
    # Text
    ro=roberta(input_ids=ids,attention_mask=am)
    ht=ro.last_hidden_state[:,0,:];ht=txt_mlp(ht)
    rtb=reg_th(ht);ctb=cls_th(ht)
    # Audio/Vision
    ha=ap(a);hao=aslstm(ha,am_a);hap,_=apool(hao['H'],am_a)
    hv=vp(v);hvo=vslstm(hv,vm_v);hvp,_=vpool(hvo['H'],vm_v)
    # AWAF
    aw=awaf(ht,hap,hvp);z=aw['Z'];w=aw['weights']
    dr=w[:,:1]*drt(ht)+w[:,1:2]*dra(hap)+w[:,2:3]*drv(hvp)
    dc=w[:,:1]*dct(ht)+w[:,1:2]*dca(hap)+w[:,2:3]*dcv(hvp)
    bdr=1.0*torch.tanh(dr);bdc=1.0*torch.tanh(dc)
    go=gate(ht,z,rtb,ctb,w,bdr);gr=go['gate_reg'];gc=go['gate_cls']
    edr=gr*dsr*bdr
    reg=rtb+edr;cls=ctb+gc*dsc*bdc
    return {'reg':reg,'cls':cls,'reg_text_base':rtb,'cls_text_base':ctb,'awaf_weights':w,'effective_delta_reg':edr,'delta_reg':dr,'residual_gate_reg':gr,'residual_gate_cls':gc},lbl

# Training loop
for epoch in range(1,EPOCHS+1):
    roberta.train();txt_mlp.train();reg_th.train();cls_th.train()
    for m in [ap,aslstm,apool,vp,vslstm,vpool,awaf,gate,drt,dra,drv,dct,dca,dcv]: m.train()
    total_loss=0.0;opt.zero_grad()
    for i,batch in enumerate(tqdm(tl,desc=f'E{epoch}',leave=False)):
        out,lbl=forward_batch(batch)
        lr=torch.nn.L1Loss()(out['reg'],lbl)
        td=lbl-out['reg_text_base'].detach()
        ld=torch.nn.SmoothL1Loss()(out['effective_delta_reg'],td)
        loss=(lr+0.2*ld)/ACCUM
        scaler.scale(loss).backward()
        if (i+1)%ACCUM==0:
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.0)
            scaler.step(opt);scaler.update();opt.zero_grad()
        total_loss+=loss.item()*ACCUM

    # Val
    roberta.eval();txt_mlp.eval();reg_th.eval();cls_th.eval()
    for m in [ap,aslstm,apool,vp,vslstm,vpool,awaf,gate,drt,dra,drv,dct,dca,dcv]: m.eval()
    vp_list,vl_list=[],[]
    with torch.no_grad():
        for batch in vl: out,lbl=forward_batch(batch);vp_list.append(out['reg'].cpu());vl_list.append(lbl.cpu())
    rp=torch.cat(vp_list);tg=torch.cat(vl_list);rs=torch.where(rp>=0,1.0,-1.0)
    m=compute_all_metrics(rp,rs,tg)
    print(f'E{epoch:2d}: loss={total_loss/len(tl):.4f} val_ACC2={m["ACC2_Non0"]:.2f}% val_MAE={m["MAE"]:.4f}')
    if m['MAE']<best_mae: best_mae=m['MAE'];best_state=True

# Test
roberta.eval();txt_mlp.eval()
for m in [reg_th,cls_th,ap,aslstm,apool,vp,vslstm,vpool,awaf,gate,drt,dra,drv,dct,dca,dcv]: m.eval()
tp_list,tl_list,aw_list,gate_list,rb_list=[],[],[],[],[]
with torch.no_grad():
    for batch in tl_test:
        out,lbl=forward_batch(batch)
        tp_list.append(out['reg'].cpu());tl_list.append(lbl.cpu())
        aw_list.append(out['awaf_weights'].cpu());gate_list.append(out['residual_gate_reg'].cpu());rb_list.append(out['reg_text_base'].cpu())

rp=torch.cat(tp_list);tg=torch.cat(tl_list);rs=torch.where(rp>=0,1.0,-1.0)
aw=torch.cat(aw_list);gv=torch.cat(gate_list);rb=torch.cat(rb_list)
m=compute_all_metrics(rp,rs,tg);mb=compute_all_metrics(rb,rs,tg)

print(f'\n=== P6G TextFT-LoRA-AWAF MOSI s{SEED} ===')
print(f'Text-base ACC2: {mb["ACC2_Non0"]:.2f}%')
print(f'Final    ACC2: {m["ACC2_Non0"]:.2f}%')
print(f'Residual GAIN: {m["ACC2_Non0"]-mb["ACC2_Non0"]:+.2f}%')
print(f'MAE: {m["MAE"]:.4f}  Corr: {m["Corr"]:.4f}  ACC7: {m["ACC7"]:.2f}%')
print(f'F1: {m["F1_Non0"]:.2f}%')
print(f'AWAF: w_t={aw[:,0].mean():.4f} w_a={aw[:,1].mean():.4f} w_v={aw[:,2].mean():.4f}')
print(f'Gate mean: {gv.mean().item():.4f}')

os.makedirs('outputs/P6G/mosi',exist_ok=True)
res={'text_base_ACC2':mb['ACC2_Non0'],'final_ACC2':m['ACC2_Non0'],'MAE':m['MAE'],'Corr':m['Corr'],'ACC7':m['ACC7'],
     'F1':m['F1_Non0'],'awaf_w_t':float(aw[:,0].mean()),'awaf_w_a':float(aw[:,1].mean()),'awaf_w_v':float(aw[:,2].mean()),
     'gate_mean':float(gv.mean()),'trainable_M':trainable/1e6}
json.dump(res,open('outputs/P6G/mosi/lora_seed42.json','w'))
print('Saved.')
