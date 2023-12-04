import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from transformers import WhisperProcessor


def get_dataloader(audio_path: str, processor: None, batch_size: int, shuffle: bool, type: str):
    print(f'dataloader: {type}')
    if type == 'whisper_openai' or type == 'whisper_faster':
        dataset = DataLoaderWhisperOpenai(audio_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    elif type == 'whisper_huggingface':
        dataset = DataLoaderWhisperHuggingface(audio_path, processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=collate_fn_whisper, num_workers=8)
    else:
        dataset = DataLoaderLS(audio_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=collate_fn, num_workers=8)

    return dataloader


class DataLoaderLS(Dataset):
    def __init__(self, dir_path):
        with open(os.path.join(dir_path, 'wav.scp'), 'r') as f:
            self.audio_file = f.readlines()

        self.text_dic = {}
        with open(os.path.join(dir_path, 'text'), 'r') as f:
            text_list = f.readlines()
            for v in text_list:
                v_split = v.split()
                idx = v_split[0]
                text = ' '.join(v_split[1:]).strip()
                self.text_dic[idx] = text

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, index):
        audio_path = self.audio_file[index].split()[1]
        wav, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resample = torchaudio.transforms.Resample(sr, 16000)
            wav = resample(wav)
            sr = 16000

        ref = self.text_dic[self.audio_file[index].split()[0]]
        text = torch.full((len(ref),), 2)
        wav_length = wav.shape[1]
        text_length = len(ref)

        return wav, sr, wav_length, text, text_length, ref


class DataLoaderWhisperOpenai(Dataset):
    def __init__(self, dir_path):
        with open(os.path.join(dir_path, 'wav.scp'), 'r') as f:
            self.audio_file = f.readlines()

        self.text_dic = {}
        with open(os.path.join(dir_path, 'text'), 'r') as f:
            text_list = f.readlines()
            for v in text_list:
                v_split = v.split()
                idx = v_split[0]
                text = ' '.join(v_split[1:]).strip()
                self.text_dic[idx] = text

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, index):
        audio_path = self.audio_file[index].split()[1]
        idx = self.audio_file[index].split()[0]
        ref = self.text_dic[idx]

        return audio_path, ref, idx


class DataLoaderWhisperHuggingface(Dataset):
    def __init__(self, dir_path, processor):
        with open(os.path.join(dir_path, 'wav.scp'), 'r') as f:
            self.audio_file = f.readlines()

        self.text_dic = {}
        self.processor = processor

        with open(os.path.join(dir_path, 'text'), 'r') as f:
            text_list = f.readlines()
            for v in text_list:
                v_split = v.split()
                idx = v_split[0]
                text = ' '.join(v_split[1:]).strip()
                self.text_dic[idx] = text

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, index):
        audio_path = self.audio_file[index].split()[1]
        wav, sr = torchaudio.load(audio_path)

        if sr != 16000:
            resample = torchaudio.transforms.Resample(sr, 16000)
            wav = resample(wav)
            sr = 16000

        features = self.processor(wav.squeeze(0), sampling_rate=sr, return_tensors="pt").input_features
        idx = self.audio_file[index].split()[0]
        ref = self.text_dic[idx]

        return features, ref, idx


def collate_fn(batch):
    wav, sr, wav_length, text, text_length, ref = zip(*batch)

    max_wav_len = max(wav_length)
    max_text_len = max(text_length)

    wav = list(wav)
    text = list(text)
    for i in range(len(wav)):
        single_len = wav[i].shape[1]
        single_wav = F.pad(wav[i], (0, max_wav_len-single_len), 'constant', 0.)
        wav[i] = single_wav

    for i in range(len(text)):

        single_len = text[i].shape[0]
        single_text = F.pad(text[i], (0, max_text_len-single_len), 'constant', 0.)
        text[i] = single_text

    wav = torch.cat(wav, dim=0)
    text = torch.stack(text, dim=0)

    return wav, torch.tensor(sr), torch.tensor(wav_length), text, torch.tensor(text_length), list(ref)


def collate_fn_whisper(batch):
    features, ref, idx = zip(*batch)

    features = torch.cat(features, dim=0)

    return features, list(ref), list(idx)


if __name__ == "__main__":
    audio_paths = '/data2/yumingdong/data/test_1000Cantonese'
    whisper_processor = WhisperProcessor.from_pretrained('/data1/yumingdong/model/huggingface_model/whisper-large-v3')
    dataset = DataLoaderWhisperHuggingface(audio_paths, whisper_processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_whisper)

    for batch in dataloader:
        for i in batch:
            print(i.shape)

