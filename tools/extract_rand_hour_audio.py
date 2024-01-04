import os
import sys
import random
from tqdm import tqdm

from utils import get_file, save_file
from tools.character_analysis import analysis

'''
extract audio with certain duration in random order from wav.scp / text 
save file: wav.scp text count.trn
Args:
    root_dir: directory which including wav.scp, text
    duration: hours
    export_dir: directory which including wav.scp, text
'''


def extract_save(root_dir: str, duration: float, tgt_dir: str):
    root_wav = get_file(os.path.join(root_dir, 'wav.scp'))
    root_txt = get_file(os.path.join(root_dir, 'text'))
    tgt_wav = []
    tgt_txt = []

    duration = duration * 3600
    idx_dict = {}
    total_duration = 0
    cnt_1_10 = 0
    cnt_10_20 = 0
    cnt_20_30 = 0
    cnt_30 = 0
    # extract -> idx_dict: {idx: [text]}
    print('Extracting...')
    pbar = tqdm(total=duration)
    random.shuffle(root_txt)
    for line in root_txt:
        item = line.split(' ')
        if len(item) < 2:
            continue

        idx = item[0]
        # if 'wenetspeech' in idx:
        #     continue
        text = ''.join(item[1:])

        bg, ed, _ = idx.split('-')[-3:]

        bg = float(bg) * 0.001
        ed = float(ed) * 0.001
        tmp = ed - bg
        # filter
        if tmp < 1:
            continue
        elif 1 <= tmp < 10:
            cnt_1_10 += 1
        elif 10 <= tmp < 20:
            cnt_10_20 += 1
        elif 20< tmp <= 30:
            cnt_20_30 += 1
        else:
            cnt_30 += 1
        total_duration += tmp
        pbar.update(tmp)
        if total_duration >= duration:
            break

        idx_dict[idx] = text

    # get corresponding audio path -> idx_dict: {idx: [text, path]}
    print('Writing...')
    for line in tqdm(root_wav):
        idx, path = line.split(' ')
        if idx in idx_dict:
            tgt_wav.append(f'{idx} {path}')
            tgt_txt.append(f'{idx} {idx_dict[idx]}')

    os.makedirs(tgt_dir, exist_ok=True)
    analysis_result = []
    # analysis_result = analysis(tgt_txt, is_wenet=True)
    analysis_result.append(f'Duration distribution:')
    analysis_result.append(f'1-10: {cnt_1_10}')
    analysis_result.append(f'10-20: {cnt_10_20}')
    analysis_result.append(f'20-30: {cnt_20_30}')
    analysis_result.append(f' > 30: {cnt_30}')
    save_file(os.path.join(tgt_dir, 'wav.scp'), tgt_wav)
    save_file(os.path.join(tgt_dir, 'text'), tgt_txt)
    save_file(os.path.join(tgt_dir, 'analysis.trn'), analysis_result)



def mian():
    root_dir = sys.argv[1]
    duration = float(sys.argv[2])
    tgt_dir = sys.argv[3]
    extract_save(root_dir, duration, tgt_dir)


if __name__ == "__main__":
    mian()

