import os
import torch
import numpy as np
import torch.nn as nn
import librosa
from transformers import (HubertForCTC,
                          Wav2Vec2ForCTC,
                          WavLMModel,
                          Data2VecAudioForCTC)
from transformers import WhisperProcessor, WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration
from transformers import MCTCTForCTC, MCTCTProcessor


class HuBERT(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.hubert_model = HubertForCTC.from_pretrained(model_path)

    def forward(self, x):
        x = self.hubert_model(x)
        return x.logits


class Wav2vec2(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_path)

    def forward(self, x):
        x = self.wav2vec_model(x)
        return x.logits


class WavLM(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.wavlm_model = WavLMModel.from_pretrained(model_path)

    def forward(self, x):
        x = self.wavlm_model(x)['last_hidden_state']
        x = x.mean(dim=1)
        return x


class Data2vec(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.data2vec = Data2VecAudioForCTC.from_pretrained(model_path)

    def forward(self, x):
        x = self.data2vec(x)['last_hidden_state']
        x = x.mean(dim=1)
        return x


class Whisper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.whisper_model.config.forced_decoder_ids = None
        self.whisper_model.config.suppress = []

    def forward(self, x, text):
        x = self.whisper_model(x, decoder_input_ids=text)
        return x


class MSS(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = MCTCTForCTC.from_pretrained(model_path)
        self.processor = MCTCTProcessor.from_pretrained(model_path)
        self.device = device
    def forward(self, x):
        x = self.processor(x, sampling_rate=16000, return_tensors='pt').input_features.to(self.device)
        logits = self.model(x).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        trans = self.processor.batch_decode(predicted_ids)
        return logits, trans
