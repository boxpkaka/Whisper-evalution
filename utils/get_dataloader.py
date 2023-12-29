import os
import librosa
import torch
from typing import Dict, Tuple, List
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from .get_save_file import get_file


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
        self.data_list = self._load_data()

    def _load_data(self) -> List:
        audio_path = os.path.join(self.data_dir, 'wav.scp')
        text_path = os.path.join(self.data_dir, 'text')
        audio_file = get_file(audio_path)
        text_file = get_file(text_path)
        idx_dic = {}
        data_list = []

        for line in text_file:
            line = line.split(' ')
            idx = line[0]
            text = ''.join(line[1:]).strip()
            idx_dic[idx] = text

        for line in audio_file:
            idx, path = line.split(' ')
            data_list.append({'idx': idx, 'audio_path': path, 'text': idx_dic[idx]})

        return data_list

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def _load_resample(audio_path: str,  tgt_sr=16000) -> Tuple:
        try:
            wav, sr = librosa.load(audio_path)
        except Exception as e:
            print(f'Error on {audio_path}: {e}')
            return None, None
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=tgt_sr)
            sr = tgt_sr
        return wav, sr


class DataLoaderPath(BaseDataloader):
    def __getitem__(self, index):
        audio_path = self.data_list[index]['audio_path']
        ref = self.data_list[index]['text']
        idx = self.data_list[index]['idx']
        return audio_path, ref, idx


class DataLoaderFeatures(BaseDataloader):
    def __init__(self, data_dir, processor):
        super().__init__(data_dir)
        self.processor = processor

    def __getitem__(self, index):
        audio_path = self.data_list[index]['audio_path']
        ref = self.data_list[index]['text']
        idx = self.data_list[index]['idx']

        wav, sr = self._load_resample(audio_path)
        if wav is None:
            return None

        if wav.shape[0] / sr > 30:
            print(f'Audio: {idx} length over 30s')
        features = self.processor(wav, sampling_rate=sr, return_tensors="pt").input_features

        return features, ref, idx


class DataLoaderDict(BaseDataloader):
    def __getitem__(self, index):
        audio_path = self.data_list[index]['audio_path']
        ref = self.data_list[index]['text']
        idx = self.data_list[index]['idx']
        wav, sr = self._load_resample(audio_path)
        if wav is None:
            return None
        if wav.shape[0] / sr > 30:
            print(f'Audio: {idx} length over 30s')
        item = {"sampling_rate": sr, "raw": wav}
        return item, ref, idx


def collate_fn_features(batch):
    batch = [item for item in batch if item is not None]
    features, ref, idx = zip(*batch)

    features = tuple(features)
    features = torch.cat(features, dim=0)
    return features, list(ref), list(idx)


def collate_fn_dict(batch):
    batch = [item for item in batch if item is not None]
    item, ref, idx = zip(*batch)

    return list(item), list(ref), list(idx)


if __name__ == "__main__":
    audio_path = '/data2/yumingdong/data/deploy_test-cantonese'
    model_path = '/data1/yumingdong/model/huggingface/whisper-small'
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    dataloader = get_dataloader(audio_path, batch_size=32, num_workers=16, shuffle=False, return_type='dict', processor=processor)

    for batch in tqdm(dataloader):
        data, ref, idx = batch
        print(data)





