"""把 adapter 产物搬运到 TMDC-main/gcnet_datasets/, 模拟 GCNet 标准布局

目标布局 (相对 TMDC-main 工作目录):
  ./gcnet_datasets/cmu_mosi_features/
      wav2vec-large-c-UTT/<uid>.npy
      deberta-large-4-UTT/<uid>.npy
      manet_UTT/<uid>.npy
  ./gcnet_datasets/CMUMOSI_features_raw_2way.pkl

TMDC 入口用 --audio-feature/--text-feature/--video-feature 三个参数指定子目录名.
shell 脚本 (run_TMDC_cmumosi.sh) 默认就用 wav2vec-large-c-UTT / deberta-large-4-UTT / manet_UTT.
"""
import os, shutil

ADAPTER = r'D:\business\pycharm\project\Tri_modal_ER/tmdc_adapter/features/mosi'
PKL_IN = os.path.join(ADAPTER, '../mosi_pkls/CMUMOSI_features_raw_2way.pkl')
DST = r'D:\business\pycharm\project\jqxxtest\TMDC-main/gcnet_datasets'

# 三个特征子目录
for sub in ['wav2vec-large-c-UTT', 'deberta-large-4-UTT', 'manet_UTT']:
    src_dir = os.path.join(ADAPTER, sub)
    dst_dir = os.path.join(DST, 'cmu_mosi_features', sub)
    os.makedirs(dst_dir, exist_ok=True)
    # 软链接或复制? Windows 软链接需要管理员, 用 robocopy / 复制
    # TMDC dataloader 是单 GPU + 小数据集, 复制即可
    for f in os.listdir(src_dir):
        s = os.path.join(src_dir, f)
        d = os.path.join(dst_dir, f)
        if not os.path.exists(d):
            shutil.copy2(s, d)
    print(f'  {sub}: 复制 {len(os.listdir(dst_dir))} 个 npy 到 {dst_dir}')

# pkl
pkl_dst = os.path.join(DST, 'CMUMOSI_features_raw_2way.pkl')
shutil.copy2(PKL_IN, pkl_dst)
print(f'  pkl 复制到 {pkl_dst}')

print('\n完成. 准备开始 TMDC 训练.')
