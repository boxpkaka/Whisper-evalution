import sys
import opencc
from jiwer import cer
from typing import List
from utils import get_file

'''
对text和转录(.trn)文本进行中文简繁统计分析(CER)
format:
    text: [idx utt]
    file.trn: [utt (idx)]
'''


def analysis(items: List[str], is_wenet: bool) -> List[str]:
    converter_tradition = opencc.OpenCC('s2t.json')
    converter_hk = opencc.OpenCC('s2hk.json')
    converter_sim = opencc.OpenCC('t2s.json')

    orig = []
    tradition = []
    hk = []
    sim = []

    for line in items:
        if len(line.split(' ')) < 2:
            print(line)
            continue
        if is_wenet:
            # Text file
            _, text = line.split(' ')
        else:
            # Transcription
            line_split = line.split(' ')
            text = ''.join(line_split[:-1])
        text_tradition = converter_tradition.convert(text)
        text_hk = converter_hk.convert(text)
        text_sim = converter_sim.convert(text)

        orig.append(text)
        tradition.append(text_tradition)
        hk.append(text_hk)
        sim.append(text_sim)

    compare_orig_tradition = cer(orig, tradition) * 100
    compare_orig_hk = cer(orig, hk) * 100
    compare_orig_sim = cer(orig, sim) * 100
    compare_tradition_hk = cer(tradition, hk) * 100

    result = []
    result.append(f'origin    -> tradition : {compare_orig_tradition}')
    result.append(f'origin    -> simplified: {compare_orig_sim}')
    result.append(f'origin    -> hk        : {compare_orig_hk}')
    result.append(f'tradition -> hk        : {compare_tradition_hk}')

    return result


if __name__ == "__main__":
    path = sys.argv[1]
    items = get_file(path)
    is_wenet = True if path.split('/')[-1] == 'text' else False
    result = analysis(items, is_wenet)
    for i in result:
        print(i)
