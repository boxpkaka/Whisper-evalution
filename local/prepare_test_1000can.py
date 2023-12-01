import os
from typing import List


def main():
    root_dir = '/data2/yumingdong/data/test_1000Cantonese/data'  # root dir -> wav.scp text text.fmt
    export_dir = '/data1/yumingdong/project/data_test_1000can'
    data_tgt_dir = '/data2/yumingdong/data/test_1000Cantonese'

    generate(root_dir, export_dir, data_tgt_dir)


def generate(root_dir: str, export_dir: str, data_tgt_dir: str):
    os.makedirs(export_dir, exist_ok=True)

    with open(os.path.join(root_dir, 'wav.scp'), 'r') as f:
        origin_wav = f.readlines()

    for i in range(len(origin_wav)):
        origin_wav_split = origin_wav[i].strip().split(' ')
        idx = origin_wav_split[0]
        origin_wav_name = origin_wav_split[1].split('/')[-1]
        new_wav_dir = os.path.join(data_tgt_dir, origin_wav_name)
        new_item = ' '.join([idx, new_wav_dir])
        print(new_item)
        origin_wav[i] = new_item

    generate_wav(origin_wav, export_dir)


def generate_wav(wav: List, export: str):
    with open(os.path.join(export, 'wav.scp'), 'w') as file:
        for item in wav:
            file.write(item + '\n')


if __name__ == '__main__':
    main()
