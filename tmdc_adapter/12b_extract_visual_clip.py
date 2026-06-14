"""MOSI 视觉特征 — CLIP-ViT-B/32 抽帧, 分批前向"""
import os, csv, time
import numpy as np
import torch
import cv2
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HOME'] = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import CLIPVisionModel, CLIPImageProcessor

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER\tmdc_adapter\features\mosi\manet_UTT'
os.makedirs(OUT, exist_ok=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = 'openai/clip-vit-base-patch32'
    proc = CLIPImageProcessor.from_pretrained(name)
    model = CLIPVisionModel.from_pretrained(name).to(device).eval().half()
    print(f'hidden_size: {model.config.hidden_size}', flush=True)

    with open(os.path.join(DATA, r'mosi\label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    mp4_root = os.path.join(DATA, r'mosi\Raw')

    # 固定随机投影
    np.random.seed(42)
    proj = np.random.randn(768, 1024).astype(np.float32) / np.sqrt(768)

    done, skipped, errors = 0, 0, 0
    t0 = time.time()
    for i, r in enumerate(rows):
        vid, cid = r['video_id'], r['clip_id']
        uid = f'{vid}_{cid}'
        out_path = os.path.join(OUT, uid + '.npy')
        if os.path.exists(out_path):
            skipped += 1
            continue
        mp4_path = os.path.join(mp4_root, vid, cid + '.mp4')
        if not os.path.exists(mp4_path):
            errors += 1
            continue
        try:
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(fps * 0.5))
            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if idx % interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
                idx += 1
            cap.release()
            if not frames:
                np.save(out_path, np.zeros((1024,), dtype=np.float32))
                skipped += 1
                continue
            # 分批, batch=4 防止 OOM
            all_feats = []
            B = 4
            for s in range(0, len(frames), B):
                batch = frames[s:s+B]
                inputs = proc(images=batch, return_tensors='pt')
                inputs = {k: v.to(device).half() for k,v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                all_feats.append(out.pooler_output.float().cpu().numpy())
            feat_768 = np.concatenate(all_feats, axis=0).mean(axis=0)
            feat = feat_768 @ proj
            np.save(out_path, feat.astype(np.float32))
            done += 1
        except Exception as e:
            errors += 1
            print(f'  err {uid}: {type(e).__name__}: {e}', flush=True)
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(rows) - i - 1) / max(rate, 0.01)
            print(f'  [{i+1}/{len(rows)}] done={done} skip={skipped} err={errors} | {rate:.2f}/s ETA {eta/60:.1f}min', flush=True)
    print(f'\n完成: done={done} skipped={skipped} errors={errors} 用时 {(time.time()-t0)/60:.1f}min', flush=True)

if __name__ == '__main__':
    main()
