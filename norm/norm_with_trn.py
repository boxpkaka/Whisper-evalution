import os
import multiprocessing
from typing import List
import opencc
from tqdm import tqdm
import re
from norm.num2char import num_to_char


def get_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    return file


def norm_single_sentence(sentence: str) -> str:
    converter = opencc.OpenCC('s2t.json')
    sentence = sentence.strip()
    # filter special symbol
    sentence = re.sub(r'[^\w\s]', '', sentence).replace(' ', '')
    # simplified -> traditional Chinese
    sentence = converter.convert(sentence).lower()
    # lower English character
    sentence = sentence.lower()
    # digit -> Chinese character
    sentence = num_to_char(sentence)
    return sentence


def single_normalization(args):
    trans, refs, start, end = args
    trans_result, refs_result = [], []
    for i in tqdm(range(start, end)):
        trans_s = trans[i].strip().split('(')
        refs_s = refs[i].strip().split('(')

        trans_idx = trans_s[1][: -1]
        refs_idx = refs_s[1][: -1]

        trans_norm = norm_single_sentence(trans_s[0])
        refs_norm = norm_single_sentence(refs_s[0])

        trans_result.append(f'{trans_norm} ({trans_idx})')
        refs_result.append(f'{refs_norm} ({refs_idx})')
    return trans_result, refs_result


def normalize_cantonese(path: str) -> None:
    trans = get_file(os.path.join(path, 'reg_orig.trn'))
    refs = get_file(os.path.join(path, 'std_orig.trn'))
    num_processes = multiprocessing.cpu_count()
    length = len(trans)
    chunk_size = length // num_processes
    process_args = []

    for i in range(num_processes):
        start = i * chunk_size
        end = length if i == num_processes - 1 else (i + 1) * chunk_size
        process_args.append((trans, refs, start, end))

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(single_normalization, process_args)

    final_trans = [item for sublist, _ in results for item in sublist]
    final_refs = [item for _, sublist in results for item in sublist]

    with open(os.path.join(path, 'reg.trn'), 'w') as file:
        for i in final_trans:
            file.write(i + '\n')

    with open(os.path.join(path, 'std.trn'), 'w') as file:
        for i in final_refs:
            file.write(i + '\n')

    print(path + ' done!')


if __name__ == '__main__':
    file_dir = '/data1/yumingdong/project/exp/openai-whisper-large-v3-data_test_hk_can'
    normalize_cantonese(file_dir)
