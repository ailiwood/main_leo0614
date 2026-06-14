"""MOSI 视觉特征 — 直接复用 Tri_modal_ER 预抽的 Frames/<vid>/<clip>.pt

Tri_modal_ER/data/mosi/Frames/<vid>/<clip>.pt 已经是 (1, 1024) 的预抽视觉特征,
与论文 MANet 1024 维对齐. 不再额外抽 (MANet 无 HF 权重, TMDC Conv1d 自适应维度).
"""
import os, csv, time
import numpy as np
import torch

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER/tmdc_adapter/features/mosi/manet_UTT'
os.makedirs(OUT, exist_ok=True)

def main():
    with open(os.path.join(DATA, 'mosi/label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    frames_root = os.path.join(DATA, 'mosi/Frames')
    done, skipped, errors = 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, cid = r['video_id'], r['clip_id']
        uid = f'{vid}_{cid}'
        out_path = os.path.join(OUT, uid + '.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        pt_path = os.path.join(frames_root, vid, cid + '.pt')
        if not os.path.exists(pt_path):
            errors += 1
            continue
        try:
            t = torch.load(pt_path, map_location='cpu', weights_only=False)
            if t.dim() == 2 and t.shape[0] == 1:
                feat = t.squeeze(0).float().numpy()
            elif t.dim() == 1:
                feat = t.float().numpy()
            else:
                feat = t.float().mean(dim=0).numpy()
            np.save(out_path, feat)
            done += 1
        except Exception as e:
            errors += 1
            print(f'  err {uid}: {e}')
        if (i+1) % 500 == 0:
            print(f'  [{i+1}/{len(rows)}] done={done} skip={skipped} err={errors}')
    print(f'\n完成: done={done} skipped={skipped} errors={errors} 用时 {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()