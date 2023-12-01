import os
from typing import List
import opencc
from jiwer import cer
import re
from norm.num2char import num_to_char


def get_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return file


def normalize_cantonese(path: str) -> None:
    converter = opencc.OpenCC('s2t.json')
    trans = get_file(os.path.join(path, 'trans'))
    refs = get_file(os.path.join(path, 'refs'))

    count = 0
    for i in range(len(trans)):
        trans_filter_symbol = re.sub(r'[^\w\s]', '', trans[i].strip()).replace(' ', '')
        trans_converted = converter.convert(trans_filter_symbol).lower()
        refs_converted = converter.convert(refs[i].strip()).lower()
        trans_converted = num_to_char(trans_converted)

        trans[i] = trans_converted
        refs[i] = refs_converted

    with open(os.path.join(path, 'reg.trn'), 'w') as file:
        for i in trans:
            file.write(i + '\n')

    with open(os.path.join(path, 'std.trn'), 'w') as file:
        for i in refs:
            file.write(i + '\n')


def compute_cer(file_dir: str) -> None:
    refs = get_file(os.path.join(file_dir, 'refs_norm'))
    trans = get_file(os.path.join(file_dir, 'trans_norm'))

    error = cer(refs, trans)
    print(error)


if __name__ == '__main__':
    file_dir = '/data1/yumingdong/project/exp/whisper-large-v2-cantonese-data_hk_can'
    normalize_cantonese(file_dir)
    compute_cer(file_dir)
