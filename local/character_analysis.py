from utils import get_file
import opencc
from jiwer import cer

converter_tradition = opencc.OpenCC('s2t.json')
converter_hk = opencc.OpenCC('s2hk.json')
converter_sim = opencc.OpenCC('t2s.json')
path = '/data2/yumingdong/data/cantonese/train/text'

lines = get_file(path)
orig = []
tradition = []
hk = []
sim = []

for line in lines:
    splited = line.split(' ')
    if len(splited) < 2:
        print(splited)
        continue
    _, text = line.split(' ')
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


