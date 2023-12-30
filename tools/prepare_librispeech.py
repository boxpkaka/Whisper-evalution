import os
import sys
from utils.get_save_file import get_file, save_file

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
    wav = []
    text = []

    for i in range(3):
        wav.extend(get_file(os.path.join(export_dir, split[i], 'wav.scp')))
        text.extend(get_file(os.path.join(export_dir, split[i], 'text')))

    save_file(os.path.join(export_dir, 'train-960', 'wav.scp'), wav)
    save_file(os.path.join(export_dir, 'train-960', 'text'), text)


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
                    text = get_file(os.path.join(dir, i))
                    save_file(export, text)

    save_file(os.path.join(export, 'wav.scp'), wav)


if __name__ == '__main__':
    main()

