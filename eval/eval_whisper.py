import torch
import whisper
import os
import time
import re
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,)
from transformers import MCTCTForCTC, MCTCTProcessor
from model import count_model
from faster_whisper import WhisperModel
from dataloader import get_dataloader
from transformers import AutoTokenizer
from eval.eval_with_trn import eval_with_trn
from eval.norm_with_trn import normalize_cantonese
import argparse

FILTER_POSTFIX = {'data': 0, 'train': 0, 'dev': 0}


def load_whisper(path: str):
    whisper = WhisperForConditionalGeneration.from_pretrained(path)
    whisper_processor = WhisperProcessor.from_pretrained(path)

    return whisper, whisper_processor


def save2export(export_dir, refs, trans):
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, 'std_orig.trn'), 'w') as f:
        for i in refs:
            f.write(i + '\n')

    with open(os.path.join(export_dir, 'reg_orig.trn'), 'w') as f:
        for i in trans:
            f.write(i + '\n')


def eval_whisper_openai(model_path: str, dataset_dir: str, export_dir: str, language: str, device: torch.device):
    dataloader = get_dataloader(audio_path=dataset_dir, batch_size=1, shuffle=False, type='whisper_openai')
    model = whisper.load_model(os.path.join(model_path, 'model.pt'), device=device)
    param = count_model(model)
    print(param)

    refs = []
    trans = []
    for batch in tqdm(dataloader):
        data_path, ref, idx = batch
        data_path = data_path[0]
        ref = ref[0]
        idx = idx[0]
        transcription = model.transcribe(data_path, language=language)['text']
        refs.append(f'{ref} ({idx})')
        trans.append(f'{transcription} ({idx})')

    save2export(export_dir, refs, trans)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def eval_faster_whisper(model_path: str, dataset_dir: str, export_dir: str, language: str, device: torch.device):
    model = WhisperModel(model_path, device='cuda', compute_type="int8_float16", device_index=7, num_workers=8)
    dataloader = get_dataloader(audio_path=dataset_dir, batch_size=1, shuffle=False, type='whisper_faster')
    # param = count_model(model)
    # print(param)
    count = 0
    refs = []
    trans = []
    start = time.time()
    for batch in dataloader:
        data_path, ref, idx = batch
        data_path = data_path[0]
        ref = ref[0]
        idx = idx[0]
        segments, info = model.transcribe(audio=data_path, language=language)
        for segment in segments:
            transcription = segment.text

        print(str(count) + '/' + str(len(dataloader)))
        print('reference: ', ref)
        print('whisper:   ', transcription)

        refs.append(f'{ref} ({idx})')
        trans.append(f'{transcription} ({idx})')
        count += 1
    end = time.time()
    print(f'Inference time: {end - start}s')
    save2export(export_dir, refs, trans)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def eval_mms(model_path: str, dataset_dir: str, export_dir: str, device: torch.device):
    model = MCTCTForCTC.from_pretrained(model_path)
    processor = MCTCTProcessor.from_pretrained(model_path)

    model.config.forced_decoder_ids = None
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, 1, False, type='whisper_huggingface')

    count = 1
    refs = []
    trans = []
    model.to(device)
    model.eval()
    for batch in dataloader:
        wav, sr, wav_length, ref, idx = batch
        wav = wav[0][:wav_length[0]]
        idx = idx[0]

        input_features = processor(wav, sampling_rate=16000, return_tensors='pt').input_features.to(device)
        logits = model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        refs.append(f'{ref[0]} ({idx})')
        trans.append(f'{transcription} ({idx})')
        print(str(count) + '/' + str(len(dataloader)))
        print('reference: ', ref[0])
        print('mms:       ', transcription)
        count += 1

    save2export(export_dir, refs, trans)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def whisper_tokenizer_test():
    text = "你好"
    audio = torch.randn(250000)
    sr = 16000
    feature_extractor = WhisperFeatureExtractor.from_pretrained("/data1/yumingdong/pretrain_model/whisper-large-v3")
    tokenizer = AutoTokenizer.from_pretrained("/data1/yumingdong/pretrain_model/whisper-large-v3", language='yue')
    # tokenizer = WhisperTokenizer.from_pretrained("/data1/yumingdong/pretrain_model/whisper-large-v3",
    #                                              language="yue", task="transcribe")
    output_feature = feature_extractor(audio, sampling_rate=sr)['input_features'][0]
    label = tokenizer(text).input_ids
    trans = tokenizer.decode(label)
    print(trans)


def eval_whisper_huggingface(model_path: str, dataset_dir: str, export_dir: str,
                             batch_size: int, language: str, device: torch.device):
    model, processor = load_whisper(model_path)
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, processor, batch_size, False, type='whisper_huggingface')
    print('=' * 100)

    refs = []
    trans = []
    model.to(device)
    model.eval()

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_features, ref, idx = batch
                input_features = input_features.to(device)

                predicted_ids = model.generate(input_features, task='transcribe', language=language)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                for i in range(len(transcription)):
                    refs.append(f'{ref[i]} ({idx[i]})')
                    trans.append(f'{transcription[i]} ({idx[i]})')

    save2export(export_dir, refs, trans)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def eval_whisper(model_index: int, model_type: str, data_index: int, language: str, batch_size: int, gpu: str):
    model_name_list = ['whisper-large-v3',
                       'whisper-large-v3-lora500-final',
                       'whisper-large-v3-lora50-14000',
                       'whisper-large-v3-lora50-14000-attn-yue',
                       'whisper-large-v3-lora50-14000-attn-none',
                       'whisper-large-v3-lora50+50-14000-attn-none']

    dataset_list = ['test_1000Cantonese',
                    'test_datatang500h',
                    'test_magicdatacantonese',
                    'test_commonvoicecantonese',
                    'test_kejiyuan',
                    'test_hk_can',
                    'dev_mandarin_2h',
                    'aishell/test',
                    'tmp/0001_1',
                    'tmp/0001_2',
                    'tmp/0002_1',
                    'tmp/0002_2']

    model_dir = os.path.join('/data1/yumingdong/model/', model_type)
    dataset_dir = '/data2/yumingdong/data'
    export_root_dir = '/data1/yumingdong/whisper/whisper-eval/exp/'

    model_name = model_name_list[model_index]
    model_path = os.path.join(model_dir, model_name)

    dataset_dir = os.path.join(dataset_dir, dataset_list[data_index])

    export_postfix = dataset_dir.split('/')[-1] if data_index != 6 else 'test_aishell'
    export_dir = os.path.join(export_root_dir, model_name + '-' + export_postfix)

    device = torch.device(f'cuda:{gpu}')

    print('=' * 100)
    print('model:    ', model_name)
    print('language: ', language)
    print('test set: ', dataset_list[data_index])
    print('export:   ', export_dir)
    print('gpu:      ', gpu)

    if re.match('openai', model_name) is not None:
        eval_whisper_openai(model_path, dataset_dir, export_dir, language, device)
    elif re.match('faster', model_name) is not None:
        eval_faster_whisper(model_path, dataset_dir, export_dir, language, device)
    elif re.match('m-ctc', model_name) is not None:
        eval_mms(model_path, dataset_dir, export_dir, device)
    else:
        eval_whisper_huggingface(model_path, dataset_dir, export_dir, batch_size, language, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--model_index', '-mi', help='index of model list', type=int)
    parser.add_argument('--model_type', '-mt', help='type of model',
                        choices=['huggingface', 'finetuned', 'openai_whisper', 'wenet_whisper'],
                        type=str)
    parser.add_argument('--data_index', '-di', help='index of dataset list', type=int)
    parser.add_argument('--language', '-l', help='language', type=str)
    parser.add_argument('--batch_size', '-b', help='batch size', type=int)
    parser.add_argument('--gpu', '-g', default=0, help='gpu id', type=str)
    args = parser.parse_args()

    eval_whisper(model_index=args.model_index,
                 model_type=args.model_type,
                 data_index=args.data_index,
                 language=args.language,
                 batch_size=args.batch_size,
                 gpu=args.gpu)

