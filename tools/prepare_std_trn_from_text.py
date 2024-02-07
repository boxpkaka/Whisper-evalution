import sys
import os   

text_dir = sys.argv[1]
file_name = sys.argv[2]

with open(os.path.join(text_dir, 'text'), 'r', encoding='utf-8') as fin:
    with open(os.path.join(text_dir, file_name), 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            idx, text = line.split(' ')
            text = text.split('|>')[-1]
            item = f'{text} ({idx})'
            fout.write(item + '\n')




