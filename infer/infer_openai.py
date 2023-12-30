import os
import torch
import whisper
from typing import List
from utils import save_file


def infer_openai(model_path: str, audio_path: List, language: str, device: torch.device):
    model = whisper.load_model(model_path, device=device)

    transcription = []
    for i in audio_path:
        print(f'-------{i}')
        transcription.append(f'-------{i}')
        result = model.transcribe(i, language=language)
        for item in result['segments']:
            bg = round(float(item['start']), 2)
            ed = round(float(item['end']), 2)
            text = item['text']
            transcription.append(f'{bg} -> {ed} {text}')

    for trans in transcription:
        print(trans)
    model_name = model_path.split('/')[-2]
    save_file(os.path.join('/data1/yumingdong/whisper/whisper-eval/exp/', f'{model_name}-yue.trn'), transcription)


if __name__ == '__main__':
    model_path = '/data1/yumingdong/model/openai/whisper-large-v3-lora50+50/model.pt'
    audio_path = ['I905910667_A00001050053062779_02062833415_15627869180_103820_4bfd.wav',
                  'I905910747_A00001050053062779_02062833415_15627869180_161117_4c51.wav',
                  'I905910669_A00001050053062779_02062833415_15627869180_104031_4bff.wav',
                  'I905910749_A00001050053062779_02062833415_15627869180_161252_4c53.wav'
                  ]
    audio_path = [os.path.join('/data1/yumingdong/test/', x) for x in audio_path]
    device = torch.device('cuda:7')

    infer_openai(model_path, audio_path, 'yue', device)
