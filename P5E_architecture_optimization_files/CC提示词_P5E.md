现在进入 P5E：UGR-AWAF Residual 架构优化与性能冲刺阶段。

【阶段背景】

P5D 代码审计已完成。当前主模型 DeepText-xLSTM-AWAF Residual 2-seed mean：

- ACC2_NZ_reg = 81.25%
- MAE = 0.8056
- Corr = 0.7489
- 稳定超过 DeepMLP text-only 80.2%
- 但未达到 82/83/85/88 目标

P5D 诊断显示：

1. residual direction 正确，但效果很弱；
2. 50.4% 样本改善，49.6% 样本被破坏；
3. weak_neg 有改善，但 strong_neg 被过度修正；
4. 当前 residual 没有显式学习 `label - text_base`；
5. sample reweight / two-stage / gate 虽已实现或准备，但仍有代码实现风险；
6. 下一阶段目标是修复实现问题，并升级为 UGR-AWAF Residual V2。

【新阶段目标】

1. 修复 P5D 代码审计发现的工程问题；
2. 接入 Uncertainty-Guided Residual Gate；
3. 接入 modality-specific delta experts；
4. 接入 delta target loss；
5. 先运行诊断 ablation，再运行 P5E V2 seed42；
6. 目标先达到 82–83%；
7. 如果 P5E 仍低于 82%，停止架构小修，转 P5F 强特征升级。

【固定路径】

项目根路径：

E:\00project_code\main_leo\new_code

基础分支：

p5d-residual-stability-performance-sprint

新建分支：

p5e-ugr-awaf-residual-optimization

Conda 环境：

E:\Anaconda3\envs\mme_xlstm_stable

【本阶段禁止事项】

1. 不冻结主模型；
2. 不写论文正式结论；
3. 不启动 MOSEI 正式训练；
4. 不提交 outputs、features、checkpoints、pth/pt/ckpt；
5. 不重新启用 text-side sLSTM 作为默认路径；
6. 不回退 P4W/P5A；
7. 不同时组合所有新模块；
8. 不把诊断 ablation 写成论文正式消融。

============================================================
第一部分：读取网页版 AI 提供的文件
============================================================

【任务 1：读取架构建议与补丁文件】

用户会提供或复制以下文件：

1. docs/模型架构优化建议06171755.md
2. docs/核心文件替换说明.md
3. models/modules/uncertainty_residual_gate.py
4. models/deeptext_xlstm_awaf_residual_v2.py
5. engine/residual_losses_v2.py
6. configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml

请先阅读，不要立即训练。

输出：

reports/P5E_ugr_awaf_residual/P5E_patch_reading_report.md

============================================================
第二部分：先修代码问题
============================================================

【任务 2：修复 attention pooling 返回表达式】

检查：

models/pooling/attention_pooling.py

若存在：

return pooled, attn if return_weights else pooled

改为：

if return_weights:
    return pooled, attn
return pooled

输出：

reports/P5E_ugr_awaf_residual/P5E_attention_pooling_fix.md

【任务 3：修复 sample reweight loss】

接入或参考：

engine/residual_losses_v2.py

要求：

1. sample reweight 必须使用 reduction='none'；
2. delta target loss 必须支持；
3. margin sign loss 必须支持；
4. group-level metrics 可输出；
5. 不再使用 scalar loss 乘 sample_w.mean 的伪重加权。

输出：

reports/P5E_ugr_awaf_residual/P5E_loss_v2_fix.md

【任务 4：修复 two-stage trainer】

检查：

scripts/train_deeptext_xlstm_awaf_residual_twostage.py

必须修复：

1. Stage 1 结束后 load Stage 1 best text state；
2. Stage 2 从最佳 text branch 开始；
3. Stage 2 训练 residual 时 text branch 必须 freeze；
4. Stage 3 若退化，必须恢复 Stage 2 best；
5. 输出每阶段 test/val 指标。

输出：

reports/P5E_ugr_awaf_residual/P5E_twostage_fix.md

【任务 5：训练脚本支持 V2 model/loss】

新增或修改：

scripts/train_deeptext_xlstm_awaf_residual_v2.py

要求：

1. 支持 DeepTextXLSTMAWAFResidualV2；
2. 支持 ResidualLossV2；
3. 支持 use_uncertainty_gate；
4. 支持 use_delta_experts；
5. 支持 delta_target_loss_weight；
6. 支持 sample_reweight_enabled；
7. 保存 predictions_test.csv；
8. 保存 text_base_delta_gate_test.csv；
9. 保存 awaf_weights_test.csv；
10. 不保存 last_model.pth；
11. 只保存 best_model.pth。

输出：

reports/P5E_ugr_awaf_residual/P5E_v2_training_entry.md

============================================================
第三部分：接入 P5E V2 模型
============================================================

【任务 6：复制/接入新模型文件】

新增：

models/modules/uncertainty_residual_gate.py
models/deeptext_xlstm_awaf_residual_v2.py
engine/residual_losses_v2.py
configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml

注意：

1. 不覆盖 P5D 主模型；
2. V2 是候选；
3. P5D 模型保留作对照；
4. V2 单元测试通过后才训练。

【任务 7：新增 V2 单元测试】

新增：

scripts/test_deeptext_xlstm_awaf_residual_v2.py

测试内容：

1. MOSI dims forward/backward；
2. MOSEI SDK dims forward/backward；
3. AWAF sum=1；
4. gate range [0,1]；
5. gate_prior range [0,1]；
6. effective_delta_reg 存在且 shape 正确；
7. delta experts 输出存在；
8. loss_v2 可计算；
9. no_residual ablation；
10. no_audio/no_vision；
11. no_awaf_mean_residual。

输出：

reports/P5E_ugr_awaf_residual/P5E_v2_unit_tests.md

============================================================
第四部分：先跑诊断 ablation
============================================================

【任务 8：P5D 诊断 ablation 30ep】

在 P5E V2 训练前，先补齐 P5D 未完成的诊断 ablation：

运行 seed=42, 30ep：

1. no_residual
2. no_audio
3. no_vision
4. no_awaf_mean_residual
5. text_slstm_on

输出：

reports/P5E_ugr_awaf_residual/P5E_diagnostic_ablation_30ep.csv
reports/P5E_ugr_awaf_residual/P5E_diagnostic_ablation_30ep_summary.md

必须回答：

1. audio 是否有效；
2. vision 是否有效；
3. AWAF 是否优于 mean residual；
4. text_slstm_on 是否继续反作用；
5. 是否建议 V2 中保留 vision。

============================================================
第五部分：V2 one-stage 训练
============================================================

【任务 9：P5E V2 seed42 60ep】

配置：

model = DeepTextXLSTMAWAFResidualV2
use_uncertainty_gate = true
use_delta_experts = true
use_bounded_delta = true
max_delta = 1.5
delta_target_loss_weight = 0.2
margin_sign_loss_weight = 0.05
sample_reweight_enabled = false
seed = 42
epochs = 60
strict protocol

输出：

reports/P5E_ugr_awaf_residual/P5E_v2_seed42_result.md

判断：

1. 如果 ACC2_NZ_reg ≥ 82.0%，补跑 seed2024；
2. 如果 ACC2_NZ_reg ≥ 83.0%，标记为强候选；
3. 如果低于 P5D mean 81.25%，停止 V2 one-stage，不补 seed；
4. 如果 MAE/Corr 明显提升但 ACC2 小幅不变，记录为候选但不冻结。

【任务 10：P5E V2 seed2024 条件补跑】

仅当 seed42 ≥ 82.0% 时执行。

输出：

reports/P5E_ugr_awaf_residual/P5E_v2_2seed_summary.csv
reports/P5E_ugr_awaf_residual/P5E_v2_2seed_summary.md

============================================================
第六部分：Two-stage + delta target
============================================================

【任务 11：V2 two-stage seed42】

仅当以下任一条件满足时执行：

1. V2 one-stage seed42 ≥ 81.25%；
2. 或 V2 one-stage MAE/Corr 明显提升。

策略：

Stage 1: text-only，目标复现 DeepMLP 80.2%
Stage 2: freeze text，训练 xLSTM/AWAF residual，使用 delta target loss
Stage 3: 小学习率 joint fine-tune，仅在 Stage2 有提升时启用

输出：

reports/P5E_ugr_awaf_residual/P5E_v2_twostage_seed42_result.md

判断：

1. 如果 Stage1 < 80.0%，先修 text branch；
2. 如果 Stage2 无提升，说明 residual 分支仍弱；
3. 如果 Stage3 退化，不启用 Stage3；
4. 如果 two-stage ≥ one-stage，补 seed2024。

============================================================
第七部分：WeakNeg reweight 条件实验
============================================================

【任务 12：weak_neg reweight 条件实验】

仅当 V2 或 two-stage 有效时执行。

配置：

sample_reweight_enabled = true
weak_neg_weight = 1.5
weak_pos_weight = 1.1
near_zero_weight = 1.0
sign_focal_enabled = false

运行 seed42 60ep。

输出：

reports/P5E_ugr_awaf_residual/P5E_weakneg_reweight_seed42_result.md

必须比较：

1. overall ACC2；
2. weak_neg ACC2；
3. weak_pos ACC2；
4. strong_neg 是否退化；
5. MAE/Corr 是否退化。

如果只提升 weak_neg 但整体下降，不采用。

============================================================
第八部分：阶段决策
============================================================

【任务 13：P5E 最佳候选选择】

候选：

1. P5D original 81.25%；
2. P5E V2 one-stage；
3. P5E V2 two-stage；
4. P5E V2 + weakneg reweight。

选择标准：

1. 2-seed mean ACC2_NZ_reg 优先；
2. MAE/Corr 不得明显下降；
3. weak_neg 改善；
4. strong_neg 不过度退化；
5. gate 不坍缩；
6. AWAF 权重仍可解释；
7. 参数量和训练成本可接受。

输出：

reports/P5E_ugr_awaf_residual/P5E_best_candidate_summary.csv
reports/P5E_ugr_awaf_residual/P5E_stage_decision.md

必须回答：

1. 是否超过 P5D 81.25；
2. 是否达到 82；
3. 是否达到 83；
4. 是否接近 85；
5. 是否仍远低于 88；
6. 是否进入 P5F 强特征升级；
7. 是否可以开始准备论文方法章草稿，但不写实验结论。

============================================================
第九部分：checkpoint 清理
============================================================

【任务 14：执行 checkpoint 清理】

用户已倾向清理探索阶段权重，但执行前必须再次生成清单：

reports/P5E_ugr_awaf_residual/P5E_checkpoint_cleanup_final_plan.csv

保留：

1. P5C seed42 best；
2. P5D seed2024 best；
3. P4T C0 ×2 reference；
4. P5E 有效候选 best。

删除：

1. P4V/P4W/P5A 等未入选探索 pth；
2. last_model；
3. epoch checkpoint。

不删除：

1. reports；
2. predictions；
3. awaf weights；
4. residual analysis；
5. configs；
6. logs。

输出：

reports/P5E_ugr_awaf_residual/P5E_checkpoint_cleanup_execution.md

============================================================
第十部分：记录与 Git
============================================================

【任务 15：更新记录】

更新：

memory.md
docs/DECISIONS.md
docs/经验总结.md
CODE_REVIEW_ENTRYPOINT.md
docs/FINAL_MODEL_SPEC_DRAFT.md

新增决策：

D0xx：P5D residual 主线稳定但性能不足，进入 P5E。  
D0xx：P5E 使用 UGR-AWAF Residual V2，显式学习 residual target。  
D0xx：AWAF 权重改为直接聚合 modality-specific delta experts，提高解释性。  
D0xx：sample reweight 必须使用 per-sample unreduced loss。  
D0xx：若 P5E 仍低于 82%，停止架构小修，进入 P5F 强特征升级。

【任务 16：Git 提交】

分支：

p5e-ugr-awaf-residual-optimization

commit message：

P5E optimize residual correction with uncertainty guided AWAF

不要提交：

outputs/
data/raw/
data/features*/
*.pth
*.pt
*.ckpt
*.csd
*.h5
*.hdf5
*.npz
*.npy
*.pkl
*.wav
*.mp4
*.zip

只提交：

models/
engine/
scripts/
configs/
reports/
docs/
memory.md
HANDOFF_PHASE_13E.md

【任务 17：输出 HANDOFF_PHASE_13E.md】

必须包含：

1. 代码修复结果；
2. V2 模型接入结果；
3. 单元测试结果；
4. diagnostic ablation 结果；
5. V2 seed42；
6. V2 seed2024，如执行；
7. two-stage 结果，如执行；
8. weakneg reweight 结果，如执行；
9. 最佳候选；
10. 是否达到 82/83/85/88；
11. checkpoint 清理执行结果；
12. 是否建议进入 P5F 强特征升级；
13. GitHub 分支和 commit；
14. 仍需用户/网页版 AI 判断的问题。

【最终汇报格式】

本阶段完成：
代码修复：
V2 模型接入：
单元测试：
diagnostic ablation：
V2 one-stage：
V2 two-stage：
weakneg reweight：
最佳候选：
是否超过 P5D 81.25：
是否达到 82/83/85/88：
checkpoint 清理：
是否建议进入 P5F 强特征升级：
GitHub 分支 / commit SHA / URL：
新增/修改文件：
HANDOFF 文件：
仍需确认的问题：
:::

现在开始执行 P5E。先修代码，再测试，再训练。不要启动 MOSEI 正式训练。
