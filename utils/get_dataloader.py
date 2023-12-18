import os
import tqdm
import librosa
import torch
import soundfile
from typing import Dict, Tuple, List

from torch.utils.data import Dataset, DataLoader
from utils.get_save_file import get_file


def get_dataloader(audio_dir: str, batch_size: int, shuffle: bool, num_workers: int, return_type: str, processor=None):
    loader_class = {
        'path': DataLoaderPath,
        'feature': DataLoaderFeatures,
        'dict': DataLoaderDict}
    dataset = loader_class[return_type](audio_dir, processor=processor)
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn_features if return_type == 'feature' else collate_fn_dict
    }
    loader = DataLoader(**dataloader_kwargs)
    return loader


class BaseDataloader(Dataset):
    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.audio_file, self.text_dic = self._load_data()

    def _load_data(self) -> Tuple[List, Dict]:
        audio_path = os.path.join(self.data_dir, 'wav.scp')
        text_path = os.path.join(self.data_dir, 'text')
        audio_file = get_file(audio_path)
        text_file = get_file(text_path)
        text_dic = {}
        for line in text_file:
            line = line.split(' ')
            idx = line[0]
            text = ''.join(line[1:]).strip()
            text_dic[idx] = text
        return audio_file, text_dic

    def __len__(self):
        return len(self.audio_file)

    @staticmethod
    def _resample(wav, sr, tgt_sr=16000):
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=tgt_sr)
            sr = tgt_sr
        return wav, sr


class DataLoaderPath(BaseDataloader):
    def __getitem__(self, index):
        idx, audio_path = self.audio_file[index].split()
        ref = self.text_dic[idx]
        return audio_path, ref, idx


class DataLoaderFeatures(BaseDataloader):
    def __init__(self, data_dir, processor):
        super().__init__(data_dir)
        self.processor = processor

    def __getitem__(self, index):
        idx, audio_path = self.audio_file[index].split(' ')
        wav, sr = librosa.load(audio_path)
        wav, sr = self._resample(wav, sr)
        if wav.shape[0] / sr > 30:
            print(f'Audio: {idx} length over 30s')
        features = self.processor(wav, sampling_rate=sr, return_tensors="pt").input_features
        ref = self.text_dic[idx]
        return features, ref, idx


class DataLoaderDict(BaseDataloader):
    def __getitem__(self, index):
        idx, audio_path = self.audio_file[index].split()
        wav, sr = librosa.load(audio_path)
        wav, sr = self._resample(wav, sr)
        if wav.shape[0] / sr > 30:
            print(f'Audio: {idx} length over 30s')
        ref = self.text_dic[idx]
        item = {"sampling_rate": sr, "raw": wav}
        return item, ref, idx


def collate_fn_features(batch):
    features = [i[0] for i in batch]
    ref = [i[1] for i in batch]
    idx = [i[2] for i in batch]
    features = tuple(features)
    features = torch.cat(features, dim=0)
    return features, list(ref), list(idx)


def collate_fn_dict(batch):
    item = [i[0] for i in batch]
    ref = [i[1] for i in batch]
    idx = [i[2] for i in batch]

    return item, ref, idx


if __name__ == "__main__":
    audio_path = '/data2/yumingdong/data/test_1000Cantonese'
    model_path = '/data1/yumingdong/model/huggingface/whisper-small'
    from transformers import WhisperProcessor
    from utils.get_audio_duration import get_duration_from_idx
    processor = WhisperProcessor.from_pretrained(model_path)
    dataloader = get_dataloader(audio_path, batch_size=32, num_workers=16, shuffle=False, return_type='feature', processor=processor)

    total = 0
    max_len = 0
    for batch in tqdm.tqdm(dataloader):
        data, ref, idx = batch

        for i in idx:
            total += get_duration_from_idx(i)
            max_len = max(max_len, get_duration_from_idx(i))
    print(f'total: {total}')
    print(max_len)



