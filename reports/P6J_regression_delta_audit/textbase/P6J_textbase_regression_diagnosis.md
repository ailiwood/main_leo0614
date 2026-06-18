# P6J Textbase Regression Diagnosis

## Results Comparison

| Source | test_ACC2 | val_ACC2 | Notes |
|---|---|---|---|
| P6H original (prior run) | **86.43%** | 87.50% | Inline script, lr=5e-5, 50ep, no early stop |
| P6I textbase recovery | 83.99% | 86.57% | Class-based, lr=5e-5, 30ep, early stop E5 |
| P6J textbase p6h_eq | 83.99% | 86.57% | Class-based, lr=5e-5, 50ep, early stop E5 |
| P6H re-run | 🔄 running | | Re-running original script to verify |

## Delta: -2.44% vs P6H original

## Potential Causes Checklist

| Factor | P6H Original | P6I/P6J | Same? |
|---|---|---|---|
| Tokenizer | roberta-large | roberta-large | ✅ |
| Label/split | formal_mode=True | formal_mode=True | ✅ |
| LoRA r/alpha/targets | 16/32/query+value | 16/32/query+value | ✅ |
| text_mlp architecture | 1024→512→256 | 1024→512→256 | ✅ |
| batch_size | 4 | 4 | ✅ |
| grad_accum | 4 | 4 | ✅ |
| lr (text) | 5e-5 | 5e-5 | ✅ |
| weight_decay | 0.03 | 0.03 | ✅ |
| seed | 42 | 42 | ✅ |
| Dropout | 0.1 | 0.1 | ✅ |
| Loss | L1 | L1 | ✅ |
| Early stopping | None | patience=10 | 🔴 Different |
| Code structure | Inline script | Class-based | 🔴 Different |
| Model init device | roberta.to(cuda) before LoRA | roberta on CPU, then to(cuda) | 🔴 Different |
| RoBERTa pooler init | Included (same warning) | Included (same warning) | ✅ |
| AMP scaler | torch.cuda.amp.GradScaler | torch.amp.GradScaler | 🔴 Different API |

## Most Likely Causes

1. **Early stopping**: P6H runs all 50 epochs, P6I/P6J stops at E15. Val plateaus after E5.
2. **AMP API change**: Different GradScaler API may affect training dynamics
3. **Model init device**: LoRA params created on CPU then moved vs created on CUDA

## Recommendation

- Wait for P6H re-run to confirm 86.43% still achievable
- If confirmed: the regression is in the class-based code, investigate init order/AMP
- If not confirmed: environment change, need to find new baseline
