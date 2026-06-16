# P4A MOSEI Readiness Audit

## 现状

| 模态 | 状态 | 数量 |
|------|:--:|------|
| Text | ✅ Labels CSV 含 text+ASR 字段 | 22799 条 |
| Audio | ✅ Audio_chunk wav 文件 | 5245 个 |
| Vision | ❌ 仅有 audio 文件，无视频帧 | 0 |
| Labels | ✅ sentiment [-3,+3] | 22699 条 |

## 可行方案

1. **用户提供 MOSEI 原始视频路径** → 用 CLIP/OpenFace 提取
2. **下载 MMSA/MLCL 标准 MOSEI 特征** → 如 MLCL README 提供下载
3. **暂用 Text+Audio 双模态** → 不得冒充三模态结果
4. **COSINE/MOSEI 公开视觉特征** → 下载 FACET/OpenFace 预提取结果

## 推荐路线

建议用户确认 MOSEI 视觉特征来源后再进入 P4B。当前 P4A MOSI 结果可作为主线。
