"""MOSI 7-tuple pkl 构造

TMDC dataloader (dataloader_cmumosi.py:22) 用 latin1 解码 7-tuple pickle:
  (videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids)
- videoIDs[vid] = list[uid], 每个 uid 对应 <feature_root>/<uid>.npy
- videoLabels[vid] = list[float in [-3, 3]]
- videoSpeakers[vid] = list[str], TMDC 用 speakermap['']=0
- trainVids/valVids/testVids = list[vid]

每个视频 1 个 clip, 即 videoIDs[vid]=[uid], videoLabels[vid]=[label].
vid 命名按 video_id.
"""
import os, csv, pickle
import numpy as np

DATA = r'D:\business\pycharm\project\Tri_modal_ER\data'
OUT = r'D:\business\pycharm\project\Tri_modal_ER/tmdc_adapter/features/mosi_pkls/CMUMOSI_features_raw_2way.pkl'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def main():
    with open(os.path.join(DATA, 'mosi/label.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    videoIDs, videoLabels, videoSpeakers, videoSentences = {}, {}, {}, {}
    trainVids, valVids, testVids = [], [], []
    for r in rows:
        vid = r['video_id']
        uid = f"{r['video_id']}_{r['clip_id']}"
        if vid not in videoIDs:
            videoIDs[vid] = [uid]
            videoLabels[vid] = [float(r['label'])]
            videoSpeakers[vid] = ['']
            videoSentences[vid] = [r['text']]
            mode = r['mode']
            if mode == 'train': trainVids.append(vid)
            elif mode == 'valid': valVids.append(vid)
            elif mode == 'test': testVids.append(vid)
        else:
            videoIDs[vid].append(uid)
            videoLabels[vid].append(float(r['label']))
            videoSpeakers[vid].append('')
            videoSentences[vid].append(r['text'])

    pickle.dump((videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids),
                 open(OUT, 'wb'), protocol=2)

    # 校验
    obj = pickle.load(open(OUT, 'rb'), encoding='latin1')
    vids, labels, spks, sents, tr, va, te = obj
    print(f'videoIDs: {len(vids)} vids')
    print(f'trainVids={len(tr)} valVids={len(va)} testVids={len(te)}')
    print(f'sample vid={list(vids.keys())[0]}, uids={vids[list(vids.keys())[0]][:2]}, labels={labels[list(vids.keys())[0]][:2]}')
    print(f'label range: {min(min(l) for l in labels.values()):.2f} ~ {max(max(l) for l in labels.values()):.2f}')
    print(f'写入: {OUT}')

if __name__ == '__main__':
    main()
