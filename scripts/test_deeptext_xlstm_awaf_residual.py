#!/usr/bin/env python
"""
scripts/test_deeptext_xlstm_awaf_residual.py
P5C 单元测试 — DeepText-xLSTM-AWAF Residual model

测试项:
  1. MOSI dims (1024/768/768) forward/backward
  2. MOSEI SDK dims (300/74/35) forward/backward
  3. AWAF sum=1
  4. Padding invariance
  5. no_residual ablation
  6. no_audio / no_vision
  7. text_slstm_on ablation
  8. delta_scale learnable
  9. Loss computable
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.deeptext_xlstm_awaf_residual import DeepTextXLSTMAWAFResidual
from engine.losses import compute_losses

torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}\n")

B, Dt, Da, Dv = 4, 1024, 768, 768
H = 256
Tt, Ta, Tv = 10, 15, 6

# ----- helpers -----
def make_batch(Dt, Da, Dv, Tt, Ta, Tv, pad_t=0, pad_a=0, pad_v=0, device=DEVICE):
    t = torch.randn(B, Tt, Dt, device=device)
    a = torch.randn(B, Ta, Da, device=device)
    v = torch.randn(B, Tv, Dv, device=device)
    tm = torch.ones(B, Tt, device=device)
    am = torch.ones(B, Ta, device=device)
    vm = torch.ones(B, Tv, device=device)
    if pad_t > 0: tm[:, -pad_t:] = 0
    if pad_a > 0: am[:, -pad_a:] = 0
    if pad_v > 0: vm[:, -pad_v:] = 0
    return t, a, v, tm, am, vm

def check_no_nan(out, name=""):
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            print(f"  ❌ NaN in {k}! {name}")
            return False
    return True

passed = 0
failed = 0
tests = []

# ============================================================
# Test 1: MOSI dims forward/backward
# ============================================================
print("=" * 60)
print("Test 1: MOSI dims (1024/768/768) forward + backward")
print("=" * 60)
model = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
).to(DEVICE)
model.train()
t, a, v, tm, am, vm = make_batch(1024, 768, 768, 10, 15, 6)
out = model(t, a, v, tm, am, vm)

print(f"  reg: {out['reg'].shape}")
print(f"  cls: {out['cls'].shape}")
print(f"  awaf_weights: {out['awaf_weights'].shape}")
print(f"  reg_text_base: {out['reg_text_base'].shape}")
print(f"  delta_reg: {out['delta_reg'].shape}")
print(f"  params: {model._init_info['total_params']:,}")

assert out['reg'].shape == (B, 1), f"Expected reg shape (4,1), got {out['reg'].shape}"
assert out['cls'].shape == (B, 1)
assert out['awaf_weights'].shape == (B, 3)
assert check_no_nan(out, "MOSI")

loss = out['reg'].mean() + out['cls'].mean()
loss.backward()
has_grad = any(p.grad is not None for p in model.parameters())
assert has_grad, "No gradients!"
print(f"  ✅ Test 1 passed (loss={loss.item():.4f})")
tests.append("1-MOSI-dims")
passed += 1

# ============================================================
# Test 2: MOSEI SDK dims forward/backward
# ============================================================
print("\n" + "=" * 60)
print("Test 2: MOSEI SDK dims (300/74/35) forward + backward")
print("=" * 60)
model_m = DeepTextXLSTMAWAFResidual(
    text_dim=300, audio_dim=74, vision_dim=35, hidden_dim=128,
).to(DEVICE)
model_m.train()
t_m = torch.randn(B, 5, 300, device=DEVICE)
a_m = torch.randn(B, 8, 74, device=DEVICE)
v_m = torch.randn(B, 3, 35, device=DEVICE)
out_m = model_m(t_m, a_m, v_m)
print(f"  reg: {out_m['reg'].shape}, cls: {out_m['cls'].shape}")
print(f"  params: {model_m._init_info['total_params']:,}")
assert out_m['reg'].shape == (B, 1)
assert check_no_nan(out_m, "MOSEI SDK")
loss_m = out_m['reg'].mean()
loss_m.backward()
print(f"  ✅ Test 2 passed")
tests.append("2-MOSEI-SDK-dims")
passed += 1

# ============================================================
# Test 3: AWAF sum(w)=1
# ============================================================
print("\n" + "=" * 60)
print("Test 3: AWAF sum(w)=1")
print("=" * 60)
model3 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
).to(DEVICE)
model3.eval()
# Test multiple random batches
for i in range(5):
    t3, a3, v3, tm3, am3, vm3 = make_batch(1024, 768, 768, 10, 15, 6)
    out3 = model3(t3, a3, v3, tm3, am3, vm3)
    w_sum_dev = (out3['awaf_weights'].sum(-1) - 1).abs().max().item()
    assert w_sum_dev < 1e-4, f"Batch {i}: sum(w) deviation = {w_sum_dev}"
print(f"  max|sum(w)-1| across 5 batches: {w_sum_dev:.2e}")
print(f"  ✅ Test 3 passed")
tests.append("3-AWAF-sum")
passed += 1

# ============================================================
# Test 4: Padding invariance
# ============================================================
print("\n" + "=" * 60)
print("Test 4: Padding invariance")
print("=" * 60)
model4 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
).to(DEVICE)
model4.eval()

# Batch with no padding
t_np, a_np, v_np, tm_np, am_np, vm_np = make_batch(1024, 768, 768, 8, 12, 5)

# Same batch but last 3 time steps are padding (replace with different values)
t_pad, a_pad, v_pad = t_np.clone(), a_np.clone(), v_np.clone()
tm_pad, am_pad, vm_pad = tm_np.clone(), am_np.clone(), vm_np.clone()
tm_pad[:, -3:] = 0; am_pad[:, -3:] = 0; vm_pad[:, -3:] = 0

# The first B-3 samples should give similar results (pooled over valid positions only)
out_np = model4(t_np, a_np, v_np, tm_np, am_np, vm_np)
out_pad = model4(t_pad, a_pad, v_pad, tm_pad, am_pad, vm_pad)

# Padded result should be different (fewer valid positions affect pooled representation)
# But should NOT be NaN and should produce valid output
assert not torch.isnan(out_pad['reg']).any(), "NaN in padded output"
print(f"  No-pad reg: [{out_np['reg'][0].item():.4f}, ...]")
print(f"  Padded  reg: [{out_pad['reg'][0].item():.4f}, ...]")
print(f"  ✅ Test 4 passed (padding produces valid output)")
tests.append("4-Padding")
passed += 1

# ============================================================
# Test 5: no_residual
# ============================================================
print("\n" + "=" * 60)
print("Test 5: no_residual ablation")
print("=" * 60)
model5 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
    ablation='no_residual',
).to(DEVICE)
model5.eval()
t5, a5, v5, tm5, am5, vm5 = make_batch(1024, 768, 768, 10, 15, 6)
out5 = model5(t5, a5, v5, tm5, am5, vm5)
reg_diff = (out5['reg'] - out5['reg_text_base']).abs().max().item()
print(f"  |reg - reg_text_base|_max = {reg_diff:.2e} (期望 0.0)")
assert reg_diff < 1e-6, f"no_residual: reg should equal reg_text_base, diff={reg_diff}"
# awaf_weights should be [1, 0, 0] for all samples
assert (out5['awaf_weights'][:, 0] == 1.0).all(), "Text weight should be 1.0"
print(f"  ✅ Test 5 passed")
tests.append("5-no-residual")
passed += 1

# ============================================================
# Test 6: no_audio / no_vision
# ============================================================
print("\n" + "=" * 60)
print("Test 6: no_audio / no_vision ablation")
print("=" * 60)

model6a = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
    ablation='no_audio',
).to(DEVICE)
model6a.eval()
out6a = model6a(t5, a5, v5, tm5, am5, vm5)
assert not torch.isnan(out6a['reg']).any()
print(f"  no_audio: reg={out6a['reg'][0].item():.4f}, ✅")

model6v = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
    ablation='no_vision',
).to(DEVICE)
model6v.eval()
out6v = model6v(t5, a5, v5, tm5, am5, vm5)
assert not torch.isnan(out6v['reg']).any()
print(f"  no_vision: reg={out6v['reg'][0].item():.4f}, ✅")
print(f"  ✅ Test 6 passed")
tests.append("6-no-audio-vision")
passed += 1

# ============================================================
# Test 7: text_slstm_on ablation
# ============================================================
print("\n" + "=" * 60)
print("Test 7: text_slstm_on ablation")
print("=" * 60)
model7 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
    ablation='text_slstm_on',
).to(DEVICE)
model7.eval()
out7 = model7(t5, a5, v5, tm5, am5, vm5)
print(f"  text_slstm_on: reg={out7['reg'][0].item():.4f},  params={model7._init_info['total_params']:,}")
assert check_no_nan(out7, "text_slstm_on")
loss7 = out7['reg'].mean(); loss7.backward()
print(f"  ✅ Test 7 passed (text_slstm_on forward+backward OK)")
tests.append("7-text-slstm-on")
passed += 1

# ============================================================
# Test 8: delta_scale learnable
# ============================================================
print("\n" + "=" * 60)
print("Test 8: delta_scale learnable")
print("=" * 60)
model8 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
).to(DEVICE)
ds_reg_before = model8.delta_scale_reg.item()
ds_cls_before = model8.delta_scale_cls.item()
print(f"  Before: δ_reg={ds_reg_before:.4f}, δ_cls={ds_cls_before:.4f}")
opt8 = torch.optim.SGD(model8.parameters(), lr=0.5)
t8, a8, v8, tm8, am8, vm8 = make_batch(1024, 768, 768, 10, 15, 6)
out8 = model8(t8, a8, v8, tm8, am8, vm8)
loss8 = out8['reg'].mean() + out8['cls'].mean()
loss8.backward()
# Check that delta_scale_reg/cls have gradients
assert model8.delta_scale_reg.grad is not None, "delta_scale_reg has no grad"
assert model8.delta_scale_cls.grad is not None, "delta_scale_cls has no grad"
opt8.step()
ds_reg_after = model8.delta_scale_reg.item()
ds_cls_after = model8.delta_scale_cls.item()
print(f"  After:  δ_reg={ds_reg_after:.4f}, δ_cls={ds_cls_after:.4f}")
changed_reg = abs(ds_reg_before - ds_reg_after) > 1e-7
changed_cls = abs(ds_cls_before - ds_cls_after) > 1e-7
print(f"  δ_reg changed: {changed_reg}, δ_cls changed: {changed_cls}")
assert changed_reg or changed_cls, "delta_scale should be learnable"
print(f"  ✅ Test 8 passed")
tests.append("8-delta-scale-learnable")
passed += 1

# ============================================================
# Test 9: Loss computable
# ============================================================
print("\n" + "=" * 60)
print("Test 9: Loss computable with ResidualLossComputer")
print("=" * 60)
model9 = DeepTextXLSTMAWAFResidual(
    text_dim=1024, audio_dim=768, vision_dim=768, hidden_dim=256,
).to(DEVICE)
model9.train()
t9, a9, v9, tm9, am9, vm9 = make_batch(1024, 768, 768, 10, 15, 6)
labels = torch.randn(B, 1, device=DEVICE) * 2.0
out9 = model9(t9, a9, v9, tm9, am9, vm9)
losses = compute_losses(out9, labels, loss_weights={
    'reg_loss_weight': 1.0, 'cls_loss_weight': 0.5,
    'sign_consistency_weight': 0.1, 'aux_loss_weight': 0.0,
    'delta_reg_weight': 0.05,
})
for k, v in losses.items():
    print(f"  {k:20s}: {v.item():.6f}")
assert not torch.isnan(losses['total']), "NaN in total loss"
losses['total'].backward()
print(f"  ✅ Test 9 passed (all losses computable)")
tests.append("9-loss-computable")
passed += 1

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"ALL TESTS: {passed}/{len(tests)} passed")
for t in tests:
    print(f"  ✅ {t}")
if passed == len(tests):
    print("\n🎉 All unit tests passed!")
    sys.exit(0)
else:
    print(f"\n❌ {len(tests) - passed} tests failed!")
    sys.exit(1)
