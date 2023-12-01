import os
import sys
from typing import List

'''
将LibriSpeech数据集转换为wav.scp, text格式
'''


def main():
    root_dir = sys.argv[1]  # e.g., '/data2/yumingdong/LibriSpeech'
    export_dir = sys.argv[2]  # e.g., '/data1/yumingdong/test/data_ls'

    split = ['train-clean-100', 'train-clean-360', 'train-other-500',
             'dev-clean', 'dev-other', 'test-clean', 'test-other']

    # generate the single split wav.scp and text
    for n in split:
        generate(data_dir=os.path.join(root_dir, n), export_dir=export_dir, split=n)

    # generate the train-960 wav.scp and text
    os.makedirs(os.path.join(export_dir, 'train-960'), exist_ok=True)
    wav = ''
    text = ''

    for i in range(3):
        with open(os.path.join(export_dir, split[i], 'wav.scp'), 'r') as temp:
            wav += (temp.read().strip() + '\n')
        with open(os.path.join(export_dir, split[i], 'text'), 'r') as temp:
            text += (temp.read().strip() + '\n')

    with open(os.path.join(export_dir, 'train-960', 'wav.scp'), 'w') as file:
        file.write(wav.strip())
    with open(os.path.join(export_dir, 'train-960', 'text'), 'w') as file:
        file.write(text.strip())


def generate(data_dir: str, export_dir: str, split: str) -> None:
    wav = []
    export = os.path.join(export_dir, split)
    print('export: ', export)
    os.makedirs(export, exist_ok=True)
    name_list_a = os.listdir(data_dir)
    for a in name_list_a:
        name_list_b = os.listdir(os.path.join(data_dir, a))
        for b in name_list_b:
            dir = os.path.join(data_dir, a, b)
            name_list = os.listdir(dir)
            for i in name_list:
                if i[-4:] == 'flac':
                    wav.append(i[:-5] + ' ' + os.path.join(dir, i))
                else:
                    generate_text(text_dir=os.path.join(dir, i), export=export)

    generate_wav(wav, export)


def generate_wav(wav: List, export: str) -> None:
    with open(os.path.join(export, 'wav.scp'), 'w') as file:
        for item in wav:
            file.write(item + '\n')


def generate_text(text_dir: str, export: str) -> None:
    with open(text_dir, 'r') as file:
        text = file.read()

    with open(os.path.join(export, 'text'), 'a') as file:
        file.write(text)


if __name__ == '__main__':
    main()

