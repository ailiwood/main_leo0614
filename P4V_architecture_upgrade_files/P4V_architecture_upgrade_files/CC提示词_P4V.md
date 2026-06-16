# CC提示词_P4V.md

现在进入 P4V：修复版 C0 可信基线重跑 + AWAF-Seq / Cross-modal Transformer 架构升级阶段。

【配套文件】

用户将提供或已经提供以下文件，请按文件内容执行：

1. `docs/代码工程评价06170218.md`
2. `docs/架构升级方案.md`
3. `models/interaction/cross_modal_transformer.py`
4. `models/pooling/attention_pooling.py`
5. `models/ours_awaf_seq_xlstm.py`
6. `configs/models/ours_awaf_seq_xlstm.yaml`
7. `scripts/test_awaf_seq_modules.py`
8. `scripts/train_awaf_seq.py`
9. `核心文件替换说明.md`

【阶段背景】

P4U.1 已完成仓库补齐与 mask 修复：

- `.gitignore` 修复，data/*.py 已进入 Git；
- `data/strong_sequence_dataset.py` 已补齐；
- `engine/strict_trainer.py` 的 evaluate() 已传入 text/audio/vision 三模态 mask；
- `check_val()` 已移除 test_loader；
- `configs/data/mosi_strong_sequence.yaml` 和 `configs/models/ours_c0_strong_sequence.yaml` 已固化；
- mask invariance、import、forward/backward、strict trainer dry-run 已通过。

注意：P4T.1 的 C0 strong sequence 结果是在 mask bug 修复前得到的，因此不能作为最终架构判断依据。P4V 必须先重跑 C0_fixed，再训练 AWAF-Seq。

【固定路径】

项目根路径：

```text
E:\00project_code\main_leo\new_code
```

基础分支：

```text
p4u1-fix-repo-data-and-mask
```

新建分支：

```text
p5v-awafseq-crossmodal-upgrade
```

固定环境：

```text
E:\Anaconda3\envs\mme_xlstm_stable
Python 3.10.20
PyTorch 2.11.0+cu128 stable
CUDA 12.8
RTX 5070 Ti
```

【本阶段目标】

1. 保存 `代码工程评价06170218.md` 和 `架构升级方案.md`；
2. 新增 AWAF-Seq 相关核心代码文件；
3. 运行 AWAF-Seq 单元测试；
4. 重跑 C0_fixed strong sequence，建立 mask 修复后的可信对照；
5. 运行 AWAF-Seq 3 epoch smoke；
6. 运行 AWAF-Seq 2 seeds × 40 epoch 初训；
7. 输出 C0_fixed vs AWAF-Seq 对比报告；
8. 更新 memory、DECISIONS、经验总结、CODE_REVIEW_ENTRYPOINT；
9. push GitHub；
10. 输出 HANDOFF_PHASE_05V.md。

【本阶段禁止事项】

1. 不覆盖 `models/ours_xlstm_fusion.py`；
2. 不删除 C0；
3. 不用 test set 选 best epoch；
4. 不跑 P5 正式消融；
5. 不把当前结果写成论文正式结论；
6. 不把 vision T=1 写成视觉时序建模；
7. 不重新接入 DEConv/CME/Data2Vec；
8. 不上传 outputs、features、checkpoints、cache。

============================================================
任务 1：创建分支并保存文档
============================================================

执行：

```bash
git checkout p4u1-fix-repo-data-and-mask
git checkout -b p5v-awafseq-crossmodal-upgrade
```

保存：

```text
docs/代码工程评价06170218.md
docs/架构升级方案.md
核心文件替换说明.md
```

输出：

```text
reports/P4V_awafseq_upgrade/P4V_docs_added.md
```

============================================================
任务 2：新增 AWAF-Seq 核心代码
============================================================

新增目录：

```text
models/interaction/
models/pooling/
```

新增文件：

```text
models/interaction/cross_modal_transformer.py
models/pooling/attention_pooling.py
models/ours_awaf_seq_xlstm.py
configs/models/ours_awaf_seq_xlstm.yaml
scripts/test_awaf_seq_modules.py
scripts/train_awaf_seq.py
```

不得覆盖：

```text
models/ours_xlstm_fusion.py
models/encoders/slstm.py
models/fusion/awaf.py
engine/strict_trainer.py
```

输出：

```text
reports/P4V_awafseq_upgrade/P4V_core_files_added.md
```

============================================================
任务 3：运行模块单元测试
============================================================

运行：

```bash
python scripts/test_awaf_seq_modules.py
```

必须验证：

1. CrossModalTransformerEncoder shape/backward；
2. MaskedAttentionPooling attention sum=1；
3. AWAF-Seq forward/backward；
4. AWAF weights sum=1；
5. padding noise invariance；
6. 无 NaN。

输出：

```text
reports/P4V_awafseq_upgrade/P4V_awaf_seq_unit_tests.md
```

============================================================
任务 4：重跑 C0_fixed strong sequence
============================================================

使用：

```text
configs/models/ours_c0_strong_sequence.yaml
configs/data/mosi_strong_sequence.yaml
engine/strict_trainer.py
```

运行：

```text
model: OursXLSTMFusion
dataset: MOSI strong sequence
epochs: 40
early_stopping_patience: 8
seeds: 42, 2024
best checkpoint: val MAE
test: final once only
```

输出目录：

```text
outputs/P4V_awafseq_upgrade/c0_fixed/<seed>/
```

汇总：

```text
reports/P4V_awafseq_upgrade/P4V_c0_fixed_results.csv
reports/P4V_awafseq_upgrade/P4V_c0_fixed_summary.md
```

必须回答：

1. mask 修复后 C0 是否高于 P4T.1；
2. seed 方差是否降低；
3. ACC2_Non0_regsign 是否达到 75+；
4. MAE/Corr 是否改善；
5. AWAF 权重是否合理。

============================================================
任务 5：AWAF-Seq 3 epoch smoke
============================================================

运行：

```bash
python scripts/train_awaf_seq.py \
  --config configs/models/ours_awaf_seq_xlstm.yaml \
  --data_config configs/data/mosi_strong_sequence.yaml \
  --seed 42 \
  --epochs 3 \
  --output_dir outputs/P4V_awafseq_upgrade/smoke_awaf_seq/seed42
```

输出：

```text
reports/P4V_awafseq_upgrade/P4V_awaf_seq_3ep_smoke.md
```

必须记录：

1. 是否可训练；
2. val metrics 是否可计算；
3. test 是否 final once；
4. AWAF 权重是否 sum=1；
5. pool attention 是否无 NaN；
6. 显存占用；
7. epoch 耗时。

============================================================
任务 6：AWAF-Seq 40 epoch 初训
============================================================

如果 smoke 通过，运行：

```text
model: OursAWAFSeqXLSTM
dataset: MOSI strong sequence
epochs: 40
early_stopping_patience: 8
seeds: 42, 2024
batch_size: 16
best checkpoint: val MAE
test: final once only
```

输出目录：

```text
outputs/P4V_awafseq_upgrade/awaf_seq/<seed>/
```

总表：

```text
reports/P4V_awafseq_upgrade/P4V_awaf_seq_results.csv
```

字段至少包括：

```text
model
seed
best_epoch_by_val
test_ACC2_Non0_regsign
test_F1_Non0_regsign
test_ACC2_Non0_cls
test_F1_Non0_cls
test_MAE
test_Corr
test_ACC7
w_t_mean
w_a_mean
w_v_mean
awaf_entropy_mean
train_time_sec
params
output_dir
```

============================================================
任务 7：C0_fixed vs AWAF-Seq 对比
============================================================

输出：

```text
reports/P4V_awafseq_upgrade/P4V_c0_vs_awaf_seq_comparison.md
```

必须比较：

1. ACC2_Non0_regsign；
2. F1_Non0_regsign；
3. MAE；
4. Corr；
5. seed 稳定性；
6. 参数量；
7. 训练时间；
8. AWAF 权重；
9. attention pooling 可解释性；
10. 是否达到 80+；
11. 是否接近 85+；
12. 是否需要补 vision T>1；
13. 是否建议继续调参。

判断标准：

- 如果 AWAF-Seq 比 C0_fixed 平均提升 ≥ 2%，且两个 seed 都不退化，可作为下一轮主模型候选；
- 如果提升 1%–2%，继续调参并补 vision T>1；
- 如果提升 < 1%，优先补 vision T>1 或重审特征；
- 如果 seed 方差大，不能进入论文主表。

============================================================
任务 8：更新项目记录
============================================================

更新：

```text
memory.md
docs/DECISIONS.md
docs/经验总结.md
CODE_REVIEW_ENTRYPOINT.md
```

新增决策：

```text
D0xx：P4V 引入 AWAF-Seq + Cross-modal Transformer 作为新候选主模型。
D0xx：C0_fixed 为 mask 修复后的可信对照基线。
D0xx：AWAF 保留为样本级可解释融合模块，不被 Transformer 替代。
D0xx：Cross-modal Transformer 负责序列级交互。
D0xx：vision T=1 仍是限制，后续需补帧级视觉序列。
```

============================================================
任务 9：Git 提交
============================================================

检查：

```bash
git status
git diff --stat
```

不得提交：

```text
outputs/
data/features*/
*.pth
*.pt
*.npz
*.npy
*.pkl
*.wav
*.mp4
```

提交：

```bash
git add .
git commit -m "P4V add AWAF-Seq cross-modal transformer model"
git push origin p5v-awafseq-crossmodal-upgrade
```

输出：

```text
reports/P4V_awafseq_upgrade/P4V_git_push_report.md
```

============================================================
任务 10：输出 handoff
============================================================

输出：

```text
HANDOFF_PHASE_05V.md
```

必须包含：

1. C0_fixed 重跑结果；
2. AWAF-Seq 新增模块；
3. 单元测试结果；
4. 3ep smoke 结果；
5. 40ep 初训结果；
6. C0_fixed vs AWAF-Seq 对比；
7. 是否达到 80+ / 是否接近 85+；
8. 是否建议补 vision T>1；
9. 是否建议继续调参；
10. GitHub 分支；
11. commit SHA；
12. 仍需用户/网页版 AI 判断的问题。

【最终汇报格式】

本阶段完成：
C0_fixed 重跑结果：
AWAF-Seq 新增模块：
单元测试结果：
3ep smoke：
AWAF-Seq 40ep 初训：
C0_fixed vs AWAF-Seq 对比：
是否达到 80+ / 是否接近 85+：
是否建议补视觉 T>1：
是否建议继续调参：
GitHub 分支 / commit SHA / URL：
新增/修改文件：
HANDOFF 文件：
仍需确认的问题：
