import os
from typing import List


if __name__ == '__main__':
    wav_path = '/data1/fangcheng1050/workspace/myproject/test/can/wavs_all'
    text_path = '/data1/fangcheng1050/workspace/myproject/test/can/rlt_can/std.trn'
    export_path = '/data1/yumingdong/project/data_fangcheng_can'

    os.makedirs(export_path, exist_ok=True)
    wav_list = os.listdir(wav_path)

    wav = []
    for file in wav_list:
        idx = file[:-4]
        file_path = os.path.join(wav_path, file)
        item = idx + ' ' + file_path
        wav.append(item)

    text = []
    with open(text_path, 'r') as f:
        origin_text = f.readlines()

    for sentence in origin_text:
        sentence_split = sentence.split('(')
        ref = sentence_split[0]
        idx = sentence_split[1].strip()[:-1]
        item = idx + ' ' + ref
        text.append(item)

    dic = {item.split()[0]: item for item in text}
    audio = []
    for i in wav:
        tgt = dic.get(i.split()[0])
        if tgt is not None:
            audio.append(i)

    with open(os.path.join(export_path, 'wav.scp'), 'w') as f:
        for i in audio:
            f.write(i + '\n')

    with open(os.path.join(export_path, 'text'), 'w') as f:
        for i in text:
            f.write(i + '\n')



