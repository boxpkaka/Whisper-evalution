import os
import re
import torch
import librosa
import soundfile
from typing import List
from transformers import AutoProcessor, WhisperForConditionalGeneration

PATTERN = re.compile(r'<\|([\d.]+)\|>(.*?)<\|([\d.]+)\|>')

def infer_hf(model_path: str, audio_path: List, device: torch.device):
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    for i in audio_path:
        wav, sr = soundfile.read(i)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        chunk = wav.shape[0] // 480000 + 1
        print(f'----------{i}: ')
        transcription = []
        for j in range(int(chunk)):
            if j == chunk - 1:
                chunked_wav = wav[j * 480000:]
            else:
                chunked_wav = wav[j * 480000: (j+1) * 480000]

            features = processor(chunked_wav, sampling_rate=sr, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features=features.to(device),
                                           return_timestamps=True)
            result = processor.batch_decode(predicted_ids, skip_special_tokens=True, decode_with_timestamps=True)[0]
            matches = PATTERN.findall(result)
            for item in matches:
                bg, text, ed = item
                bg = round(float(bg) + j * 30, 2)
                ed = round(float(ed) + j * 30, 2)
                transcription.append(f'{bg} -> {ed} {text}')
        for trans in transcription:
            print(trans)


if __name__ == '__main__':
    model_path = '/data1/yumingdong/model/finetuned/whisper-large-v3-lora700+700-70000'
    audio_path = ['I905910667_A00001050053062779_02062833415_15627869180_103820_4bfd.wav',
                  'I905910747_A00001050053062779_02062833415_15627869180_161117_4c51.wav',
                  'I905910669_A00001050053062779_02062833415_15627869180_104031_4bff.wav',
                  'I905910749_A00001050053062779_02062833415_15627869180_161252_4c53.wav'
                  ]
    audio_path = [os.path.join('/data1/yumingdong/test/', x) for x in audio_path]
    device = torch.device('cuda:7')

    infer_hf(model_path, audio_path, device)
