import os
from typing import List
import opencc
import re

def get_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return file

def normalize_cantonese(text: List) -> List:
    converter = opencc.OpenCC('s2t.json')

    for i in range(len(text)):
        filter_symbol = re.sub(r'[^\w\s]', '', text[i].strip())
        print(filter_symbol)

    return []


def compute_cer(file_dir: str) -> None:
    refs = get_file(os.path.join(file_dir, 'refs'))
    trans = get_file(os.path.join(file_dir, 'trans'))

    normalize_cantonese(trans)

    # for i in range(len(refs)):
    #     print(refs[i].strip())
    #     print(trans[i].strip())


if __name__ == '__main__':
    file_dir = '/data1/yumingdong/test/exp/whisper-large-v2'
    compute_cer(file_dir)
