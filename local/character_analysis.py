import sys
import opencc
from jiwer import cer
from utils import get_file

'''
对text和转录(.trn)文本进行中文简繁统计分析(CER)
format:
    text: idx utt
    file.trn: utt (idx)
'''


def analysis(text_path: str):
    converter_tradition = opencc.OpenCC('s2t.json')
    converter_hk = opencc.OpenCC('s2hk.json')
    converter_sim = opencc.OpenCC('t2s.json')

    lines = get_file(text_path)
    orig = []
    tradition = []
    hk = []
    sim = []

    for line in lines:
        if len(line.split(' ')) < 2:
            print(line)
            continue
        if text_path.split('/')[-1] == 'text':
            # Text file
            _, text = line.split(' ')
        else:
            # Transcription
            text, _ = line.split(' ')
        text_tradition = converter_tradition.convert(text)
        text_hk = converter_hk.convert(text)
        text_sim = converter_sim.convert(text)

        orig.append(text)
        tradition.append(text_tradition)
        hk.append(text_hk)
        sim.append(text_sim)

    compare_orig_tradition = cer(orig, tradition)
    compare_orig_hk = cer(orig, hk)
    compare_orig_sim = cer(orig, sim)
    compare_tradition_hk = cer(tradition, hk)

    print(f'origin    -> tradition : {compare_orig_tradition}')
    print(f'origin    -> simplified: {compare_orig_hk}')
    print(f'origin    -> hk        : {compare_orig_sim}')
    print(f'tradition -> hk        : {compare_tradition_hk}')


if __name__ == "__main__":
    path = sys.argv[1]
    analysis(text_path=path)

