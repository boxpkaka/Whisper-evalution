import os
import sys
import soundfile
from tqdm import tqdm
from typing import List, Dict

'''
将aishell数据集转换为wav.scp, text格式
'''


def main():
    root_dir = sys.argv[1]    # Path of 'data_aishell' directory
    export_dir = sys.argv[2]  # export_dir/wav.scp|text

    split = ['train', 'dev', 'test']
    with open(os.path.join(root_dir, 'transcript', 'aishell_transcript_v0.8.txt'), 'r') as f:
        file = f.readlines()

    # {audio id:transcription}
    idx_text = {}
    for line in file:
        s = line.strip().split(' ')
        idx = s[0]
        trans = ''.join(s[1:])
        idx_text[idx] = trans

    for n in split:
        save_split_wav_text(root_dir, n, export_dir, idx_text)


def save_split_wav_text(root_dir: str, split: str, export_dir: str, idx_text: Dict) -> None:
    audio_root_dir = os.path.join(root_dir, 'wav', split)
    sub_dir_list = os.listdir(audio_root_dir)

    wav = []
    text = []
    for n in sub_dir_list:
        audio_dir = os.path.join(audio_root_dir, n)
        audio_list = os.listdir(audio_dir)

        for audio_name in audio_list:
            idx = audio_name[:-4]
            if idx_text.get(idx) is not None:
                audio_path = os.path.join(audio_dir, audio_name)
                sample, sr = soundfile.read(audio_path)
                duration = int(round(sample.shape[0] / sr, 3) * 1000)
                zero_num = 7 - len(str(duration))
                duration = zero_num * '0' + str(duration)
                wav_item = f'{idx}-0000000-{duration}-C {audio_path}'
                text_item = f'{idx}-0000000-{duration}-C {idx_text[idx]}'
                wav.append(wav_item)
                text.append(text_item)

    export_dir = os.path.join(export_dir, split)
    wav_export_path = os.path.join(export_dir, 'wav.scp')
    text_export_path = os.path.join(export_dir, 'text')
    os.makedirs(export_dir, exist_ok=True)

    save_file(wav_export_path, wav)
    save_file(text_export_path, text)


def save_file(export_path: str, file: List) -> None:
    with open(export_path, 'w') as f:
        for item in file:
            f.write(item + '\n')


if __name__ == '__main__':
    main()



