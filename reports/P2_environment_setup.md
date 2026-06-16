# P2 环境搭建报告

> 生成时间：2026-06-16  
> 环境：mme_xlstm

---

## 一、环境创建

| 项目 | 值 |
|------|-----|
| 环境名 | mme_xlstm |
| 路径 | E:\Anaconda3\envs\mme_xlstm |
| Python | 3.10.20 |
| 创建命令 | `conda create -p E:/Anaconda3/envs/mme_xlstm python=3.10 -y --override-channels -c defaults` |
| 状态 | ✅ 创建成功 |

## 二、关键依赖

| 包 | 版本 | 用途 |
|----|------|------|
| **torch** | **2.12.0.dev20260408+cu128 (GPU版, nightly)** | 深度学习框架。⚠ **临时 nightly 方案**：因 RTX 5070 Ti (Blackwell sm_120) 在 stable PyTorch 2.6.0 预编译二进制中缺少 sm_120 kernel，暂用 nightly。P4 正式训练前如 stable 版本已支持 Blackwell，应切换回 stable release |
| torchvision | 0.21.0+cu124 (GPU版) | ⚠ 与 torch nightly 版本不完全匹配，但 import 可用 |
| torchaudio | 2.6.0+cu124 (GPU版) | ⚠ 同上 |
| transformers | 4.34.1 | HuggingFace 模型加载 |
| numpy | 1.26.4 | 数值计算（降级到 1.x 避免兼容警告） |
| tqdm | 4.68.2 | 进度条 |
| einops | 0.8.2 | 张量操作 |
| scikit-learn | 1.7.2 | 指标计算辅助 |
| matplotlib | 3.10.9 | 图像绘制 |
| openpyxl | 3.1.5 | Excel 输出 |
| pyyaml | 6.0.3 | YAML 配置解析 |

## 三、踩坑记录

1. **清华镜像 SSL 问题**：conda 和 pip 默认使用 tuna.tsinghua.edu.cn 镜像，SSL 证书错误。解决：conda 用 `--override-channels -c defaults`，pip 用 `--index-url https://pypi.org/simple/`

2. **RTX 5070 Ti (Blackwell sm_120) CUDA 兼容性**：
   - PyTorch 2.3.0+cu118 不支持 Blackwell（最高支持 sm_90 Hopper）
   - 必须 PyTorch 2.5.1+cu124（CUDA 12.4 toolkit）
   - 第一次安装 2.3.0+cu118 后 GPU 不可用（警告后回退 CPU）

3. **numpy 2.x 兼容性**：torch 2.5.1 有 numpy 2.x 警告。降级到 numpy 1.26.4 解决。

4. **pip install 双 --index-url**：pip 只使用最后一个 `--index-url`，额外的源需用 `--extra-index-url`

## 四、环境文件

- `env/pip_freeze_mme_xlstm.txt` ✅ 已导出
- `env/environment_mme_xlstm.yml` ⚠ 待 conda env export 完成
- `env/install_commands.md` ✅ 已记录
