import os
import re
import sys
import torch
import librosa
import soundfile
from typing import List
from model.get_model import load_hf_whisper, load_hf_processor

PATTERN = re.compile(r'<\|([\d.]+)\|>(.*?)<\|([\d.]+)\|>')


def infer_hf(
        model_path: str,
        audio_path: List,
        device: torch.device,
        timestamp=False,
        language=None) -> None:

    model = load_hf_whisper(path=model_path, use_flash_attention_2=False, torch_dtype=torch.float32).to(device)
    processor = load_hf_processor(model_path)

    for i in audio_path:
        wav, sr = soundfile.read(i)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        chunk = wav.shape[0] // 480000 + 1
        print(f'----------{i}: ')
        transcription = []
        generate_kwargs = {
            'return_timestamps': timestamp
        }
        if language:
            generate_kwargs['language'] = language

        for j in range(int(chunk)):
            if j == chunk - 1:
                chunked_wav = wav[j * 480000:]
            else:
                chunked_wav = wav[j * 480000: (j+1) * 480000]

            features = processor(chunked_wav, sampling_rate=sr, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features=features.to(device), **generate_kwargs)
            result = processor.batch_decode(predicted_ids, skip_special_tokens=False, decode_with_timestamps=False)[0]
            matches = PATTERN.findall(result)
            for item in matches:
                bg, text, ed = item
                bg = round(float(bg) + j * 30, 2)
                ed = round(float(ed) + j * 30, 2)
                transcription.append(f'{bg} -> {ed} {text}')
        for trans in transcription:
            print(trans)


if __name__ == '__main__':
    model_dir = sys.argv[1]
    # model_path = '/data1/yumingdong/model/finetuned/whisper-large-v3-lora700+700-130000'
    audio_path = sys.argv[2]

    device = torch.device('cuda:0')

    infer_hf(model_dir, [audio_path], device, timestamp=False, language='zh')
