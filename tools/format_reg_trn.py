import sys
import os   


'''
格式化reg.trn
quanzhouyancao00000174-0009560-0010340-B-0000030-0000880-S -> quanzhouyancao00000174-0009560-0010340-B
'''

text_path = sys.argv[1]
out_path = sys.argv[2]

with open(text_path, 'r', encoding='utf-8') as fin:
    with open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            text, idx = line.split('(')
            idx = idx.split('-')
            idx = '-'.join(idx[:-3])
            item = f'{text}({idx})'
            fout.write(item + '\n')




