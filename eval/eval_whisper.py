import torch
import whisper
import os
import time
import pynvml
import psutil
from tqdm import tqdm
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from transformers import MCTCTForCTC, MCTCTProcessor
from faster_whisper import WhisperModel
from dataloader import get_dataloader
from eval.eval_with_trn import eval_with_trn
from norm.norm_with_trn import normalize_cantonese
from utils.count_model import count_model
from utils.get_save_file import get_file, save_file
from utils.count_time import CountTime
from utils.get_audio_duration import get_duration_from_idx


FILTER_POSTFIX = {'data': 0, 'train': 0, 'dev': 0}


def load_whisper(path: str):
    whisper = WhisperForConditionalGeneration.from_pretrained(path)
    whisper_processor = WhisperProcessor.from_pretrained(path)

    return whisper, whisper_processor


def save_eval(export_dir, refs, trans, trans_with_time=None):
    os.makedirs(export_dir, exist_ok=True)
    save_file(os.path.join(export_dir, 'std_orig.trn'), refs)
    save_file(os.path.join(export_dir, 'reg_orig.trn'), trans)
    if trans_with_time is not None:
        save_file(os.path.join(export_dir, 'reg_rtf.trn'), trans_with_time)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def eval_whisper_openai(model_path: str, dataset_dir: str, export_dir: str, language: str, device: torch.device):
    dataloader = get_dataloader(dataset_dir, None, batch_size=1, shuffle=False, type='whisper_openai')
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

    save_eval(export_dir, refs, trans)


def eval_faster_whisper(model_path: str, dataset_dir: str, export_dir: str, language: str, device: torch.device):
    model = WhisperModel(model_path, device='cuda', compute_type="int8_float16", device_index=7, num_workers=8)
    dataloader = get_dataloader(dataset_dir, None, batch_size=1, shuffle=False, type='whisper_faster')
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
    save_eval(export_dir, refs, trans)


def eval_mms(model_path: str, dataset_dir: str, export_dir: str, device: torch.device):
    model = MCTCTForCTC.from_pretrained(model_path)
    processor = MCTCTProcessor.from_pretrained(model_path)

    model.config.forced_decoder_ids = None
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, None, 1, False, type='whisper_huggingface')

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

    save_eval(export_dir, refs, trans)


def eval_whisper_huggingface(model_path: str, dataset_dir: str, export_dir: str,
                             batch_size: int, language: str, device: torch.device):
    model, processor = load_whisper(model_path)
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, processor, batch_size, shuffle=False, type='whisper_huggingface')
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    refs = []
    trans = []
    memory = []
    trans_with_info = []

    total_cost_time = 0
    total_audio_time = 0
    max_cpu_usage = 0

    model.to(device)
    model.eval()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for batch in tqdm(dataloader):

                with CountTime(handle) as ct:
                    input_features, ref, idx = batch
                    input_features = input_features.to(device)

                    predicted_ids = model.generate(input_features, task='transcribe', language=language)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                cost_time = ct.cost_time
                memory_used = ct.cost_memory

                total_cost_time += cost_time
                memory.append(memory_used)

                cpu_usage = psutil.cpu_percent(interval=1)
                max_cpu_usage = max(cpu_usage, max_cpu_usage)

                for i in range(len(transcription)):
                    total_audio_time += get_duration_from_idx(idx[i])
                    refs.append(f'{ref[i]} ({idx[i]})')
                    trans.append(f'{transcription[i]} ({idx[i]})')
                    if i == 0:
                        trans_with_info.append(f'batch-info: cost time: {cost_time} used memory: {memory_used} '
                                               f'cpu usage: {cpu_usage}')
                        trans_with_info.append(f'{transcription[i]} ({idx[i]}) ')
                    else:
                        trans_with_info.append(f'{transcription[i]} ({idx[i]})')

    rtf = round(total_cost_time / total_audio_time, 3)
    memory_max = max(memory)
    memory_avg = sum(memory) / len(memory)

    trans_with_info.append(f'total cost time: {total_cost_time}s')
    trans_with_info.append(f'RTF:             {rtf}')
    trans_with_info.append(f'Throughput:      {round(1/rtf, 3)}')
    trans_with_info.append(f'Avg memory:      {memory_avg}')
    trans_with_info.append(f'Max memory:      {memory_max}')
    trans_with_info.append(f'Max cpu usage:   {max_cpu_usage}')
    save_eval(export_dir, refs, trans, trans_with_info)

