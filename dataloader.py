import librosa
import torch
import soundfile
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os


def get_dataloader(audio_path: str, batch_size: int, shuffle: bool, type: str, processor=None):
    if type == 'path':
        dataset = DataLoaderAudioPath(audio_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16)
    elif type == 'feature':
        dataset = DataLoaderFeatures(audio_path, processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=collate_fn_features, num_workers=16)
    elif type == 'dict':
        dataset = DataLoaderDict(audio_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=collate_fn_dict, num_workers=16)
    return dataloader


class DataLoaderAudioPath(Dataset):
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


class DataLoaderFeatures(Dataset):
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


class DataLoaderDict(Dataset):
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
        wav, sr = soundfile.read(audio_path)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        item = {"sampling_rate": sr, "raw": wav}
        idx = self.audio_file[index].split()[0]
        ref = self.text_dic[idx]
        return item, ref, idx

def collate_fn_features(batch):
    features, ref, idx = zip(*batch)
    features = torch.cat(features, dim=0)
    return features, list(ref), list(idx)

def collate_fn_dict(batch):
    item, ref, idx = zip(*batch)
    return list(item), list(ref), list(idx)


if __name__ == "__main__":
    audio_paths = '/data2/yumingdong/data/test_1000Cantonese'
    dataloader = get_dataloader(audio_paths, batch_size=1, shuffle=False, type='dict')

    for batch in dataloader:
        for i in batch:
            print(i)

