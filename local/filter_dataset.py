import os
from utils import get_file, save_file

'''
过滤wav.scp/text文件中的无标签条目
'''


def filter(data_dir: str) -> None:
    wav_file = get_file(os.path.join(data_dir, 'wav.scp'))
    text_file = get_file(os.path.join(data_dir, 'text'))
    res = {}

    for line in wav_file:
        if len(line.split(' ')) < 2:
            continue

        idx, path = line.split(' ')
        res[idx] = [path]

    for line in text_file:
        if len(line.split(' ')) < 2:
            continue

        idx, text = line.split(' ')
        res[idx].append(text)

    wav_save = []
    text_save = []
    for idx, item in res.items():
        if len(item) < 2:
            continue
        wav_save.append(f'{idx} {item[0]}')
        text_save.append(f'{idx} {item[1]}')

    save_file(os.path.join(data_dir, 'wav.scp'), wav_save)
    save_file(os.path.join(data_dir, 'text'), text_save)


if __name__ == '__main__':
    data_dir = '/data2/yumingdong/data/cantonese/train'
    filter(data_dir)


