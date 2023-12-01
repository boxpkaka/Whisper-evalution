import os.path
from tqdm import tqdm
from pydub import AudioSegment
from typing import List, Dict


def save_segment(audio_dir: str, audio_name: str, idx_time: Dict) -> None:
    sound = AudioSegment.from_wav(os.path.join(audio_dir, audio_name))
    for k, v in tqdm(idx_time.items()):
        if k.split('-')[0] == audio_name[:-4]:
            part = sound[v[0]:v[1]]
            export_path = os.path.join(audio_dir, f'{k}.wav')
            print(export_path)
            part.export(export_path, format='wav')


audio_dir = '/data2/yumingdong/wavs/tmp'
std_dir = '/data2/yumingdong/data/tmp/'

with open(os.path.join(std_dir, 'std.trn'), 'r') as f:
    file = f.readlines()

idx_text = {}
idx_time = {}

for line in file:
    line = line.strip().split('(')
    if line[0] != '':
        idx = line[1][:-1]
        _, begin, end, _ = idx.split('-')
        text = ''.join(line[0].split(' '))
        if len(text) > 0:
            idx_text[idx] = text
            idx_time[idx] = [int(begin), int(end)]

audio_name_list = os.listdir(audio_dir)

# for n in audio_name_list:
#     save_segment(audio_dir, n, idx_time)

for n in ['0001_1', '0001_2', '0002_1', '0002_2']:
    audio_name = n
    print(audio_name)
    export_dir = os.path.join(std_dir, audio_name)
    os.makedirs(export_dir, exist_ok=True)

    with open(os.path.join(export_dir, 'text'), 'w') as f:
        for k, v in idx_text.items():
            if k.split('-')[0] == audio_name:
                item = f'{k} {v}'
                f.write(item + '\n')

    with open(os.path.join(export_dir, 'wav.scp'), 'w') as f:
        for k, v in idx_text.items():
            if k.split('-')[0] == audio_name:
                audio_path = os.path.join(audio_dir, f'{k}.wav')
                item = f'{k} {audio_path}'
                f.write(item + '\n')



