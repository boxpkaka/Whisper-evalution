import os
import re
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
        timestamp: bool,
        language=None) -> None:

    model = load_hf_whisper(path=model_path, use_flash_attention_2=True, torch_dtype=torch.bfloat16).to(device)
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

            features = processor(chunked_wav, sampling_rate=sr, return_tensors="pt").input_features.to(torch.bfloat16)
            predicted_ids = model.generate(input_features=features.to(device), **generate_kwargs)
            print(predicted_ids)
            result = processor.batch_decode(predicted_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]
            print(result)
            matches = PATTERN.findall(result)
            for item in matches:
                bg, text, ed = item
                bg = round(float(bg) + j * 30, 2)
                ed = round(float(ed) + j * 30, 2)
                transcription.append(f'{bg} -> {ed} {text}')
        for trans in transcription:
            print(trans)


if __name__ == '__main__':
    model_path = '/data1/yumingdong/whisper/Whisper-Finetune/models/whisper-large-v3-finetune'
    # model_path = '/data1/yumingdong/model/huggingface/whisper-large-v3'
    audio_path = ['/data2/yumingdong/wavs/cantonese/wavs/dev/datatang500h00000818-0754561-0756330-C1.wav',
                  '/data2/yumingdong/wavs/cantonese/wavs/dev/datatang500h00000822-0678688-0690173-C0.wav',
                  '/data2/yumingdong/wavs/data_aishell/data_aishell/wav/train/S0057/BAC009S0057W0495.wav',
                  '/data2/yumingdong/wavs/data_aishell/data_aishell/wav/train/S0057/BAC009S0057W0430.wav']

    device = torch.device('cuda:0')

    infer_hf(model_path, audio_path, device, timestamp=False)
