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
                                           return_timestamps=True,
                                           language='cantonese')
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
    model_path = '/data1/yumingdong/model/huggingface/whisper-large-v3'

    audio_path = [
        '/data2/yumingdong/wavs/test_1000Cantonese/1000Cantonese00000334-0000000-0003894-S.wav',
        '/data2/yumingdong/wavs/test_1000Cantonese/1000Cantonese00000668-0000000-0002964-S.wav'
    ]
    device = torch.device('cuda:7')

    infer_hf(model_path, audio_path, device)
