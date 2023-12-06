import torch
import whisper
import os
import pynvml
import psutil
from tqdm import tqdm
from typing import List
from transformers import MCTCTForCTC, MCTCTProcessor, WhisperProcessor
from faster_whisper import WhisperModel
from dataloader import get_dataloader
from eval.eval_with_trn import eval_with_trn
from norm.norm_with_trn import normalize_cantonese
from utils.count_model import count_model
from utils.get_save_file import save_file
from utils.count_time import CountTime
from utils.get_audio_duration import get_duration_from_idx
from utils.get_model import get_pipeline, load_whisper


def save_eval(export_dir, refs, trans, trans_with_time=None):
    os.makedirs(export_dir, exist_ok=True)
    save_file(os.path.join(export_dir, 'std_orig.trn'), refs)
    save_file(os.path.join(export_dir, 'reg_orig.trn'), trans)
    if trans_with_time is not None:
        save_file(os.path.join(export_dir, 'reg_rtf.trn'), trans_with_time)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def get_usage_info(total_cost_time: float, total_audio_time: float,
                   memory: List, max_cpu_usage: float,
                   trans_with_info: List) -> None:
    rtf = round(total_cost_time / total_audio_time, 3)
    memory_max = max(memory)
    memory_avg = sum(memory) / len(memory)

    trans_with_info.append(f'total cost time: {total_cost_time}s')
    trans_with_info.append(f'RTF:             {rtf}')
    trans_with_info.append(f'Throughput:      {round(1/rtf, 3)}')
    trans_with_info.append(f'Avg memory:      {memory_avg}')
    trans_with_info.append(f'Max memory:      {memory_max}')
    trans_with_info.append(f'Max cpu usage:   {max_cpu_usage}')


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


def eval_faster_whisper(model_path: str, dataset_dir: str, export_dir: str, language: str,
                        use_cpu: bool, int8: bool, num_workers: int,
                        device: torch.device):
    if use_cpu:
        device = 'cpu'
        compute_type = 'int8'
        device_index = None
    else:
        device = 'cuda'
        device_index = device.index
        if int8:
            compute_type = 'int8_float16'
        else:
            compute_type = 'float16'
    print(type(device_index))
    print(use_cpu)
    model = WhisperModel(model_path, device=device,
                         compute_type=compute_type, device_index=device_index, num_workers=num_workers)

    dataloader = get_dataloader(dataset_dir, None, batch_size=1, shuffle=False, type='whisper_faster')

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    refs = []
    trans = []
    memory = []
    trans_with_info = []

    total_cost_time = 0
    total_audio_time = 0
    max_cpu_usage = 0

    for batch in tqdm(dataloader):
        with CountTime(handle) as ct:
            data_path, ref, idx = batch
            data_path = data_path[0]
            ref = ref[0]
            idx = idx[0]
            segments, info = model.transcribe(audio=data_path, language=language)
            for segment in segments:
                transcription = segment.text

        cost_time = ct.cost_time
        memory = ct.cost_memory
        memory_used = ct.cost_memory

        total_cost_time += cost_time
        memory.append(memory_used)

        cpu_usage = psutil.cpu_percent(interval=1)
        max_cpu_usage = max(cpu_usage, max_cpu_usage)


        refs.append(f'{ref} ({idx})')
        trans.append(f'{transcription} ({idx})')
        trans_with_info.append(f'{transcription} ({idx})'
                               f'batch-info: cost time: {cost_time} '
                               f'used memory: {memory_used} '
                               f'cpu usage: {cpu_usage}')

    get_usage_info(total_cost_time, total_audio_time, memory, max_cpu_usage, trans_with_info)
    export_dir += f'-{device}-{compute_type}'
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
                        trans_with_info.append(f'batch-info: cost time: {cost_time} '
                                               f'used memory: {memory_used} '
                                               f'cpu usage: {cpu_usage}')
                        trans_with_info.append(f'{transcription[i]} ({idx[i]}) ')
                    else:
                        trans_with_info.append(f'{transcription[i]} ({idx[i]})')

    get_usage_info(total_cost_time, total_audio_time, memory, max_cpu_usage, trans_with_info)
    save_eval(export_dir, refs, trans, trans_with_info)


def eval_whisper_pipeline(model_path: str, dataset_dir: str, export_dir: str,
                          batch_size: int, language: str, device: torch.device):
    pipe = get_pipeline(model_path, batch_size, gpu=str(device.index))
    processor = WhisperProcessor.from_pretrained(model_path)
    generate_kwargs = {"task": 'transcribe', "num_beams": 1, "language": language}

    dataloader = get_dataloader(dataset_dir, processor, batch_size, shuffle=False, type='whisper_openai')
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

    for batch in tqdm(dataloader):
        data_path, ref, idx = batch
        data_path = list(data_path)

        with CountTime(handle) as ct:
            result = pipe(data_path, return_timestamps=False, generate_kwargs=generate_kwargs)

        cost_time = ct.cost_time
        memory_used = ct.cost_memory
        print(cost_time)

        total_cost_time += cost_time
        memory.append(memory_used)

        cpu_usage = psutil.cpu_percent(interval=1)
        max_cpu_usage = max(cpu_usage, max_cpu_usage)

        for i in range(len(result)):
            transcription = result[i]['text']
            total_audio_time += get_duration_from_idx(idx[i])
            refs.append(f'{ref[i]} ({idx[i]})')
            trans.append(f'{transcription} ({idx[i]})')
            if i == 0:
                trans_with_info.append(f'batch-info: cost time: {cost_time} '
                                       f'used memory: {memory_used} '
                                       f'cpu usage: {cpu_usage}')
                trans_with_info.append(f'{transcription} ({idx[i]}) ')
            else:
                trans_with_info.append(f'{transcription} ({idx[i]})')

    export_dir += f'-pipeline'
    get_usage_info(total_cost_time, total_audio_time, memory, max_cpu_usage, trans_with_info)
    save_eval(export_dir, refs, trans, trans_with_info)
