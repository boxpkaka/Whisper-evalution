import soundfile
import torch
import whisper
import os
import pynvml
import psutil
from tqdm import tqdm
from transformers import WhisperProcessor
from faster_whisper import WhisperModel
from dataloader import get_dataloader
from eval.eval_with_trn import eval_with_trn
from norm.norm_with_trn import normalize_cantonese
from utils.count_model import count_model
from utils.get_save_file import save_file
from utils.count_usage import StepCounter, TrainMonitor
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
                        use_cpu: bool, int8: bool, num_workers: int, device: torch.device):
    if use_cpu:
        device = 'cpu'
        compute_type = 'int8'
        device_index = None
    else:
        device_index = device.index
        device = 'cuda'
        if int8:
            compute_type = 'int8_float16'
        else:
            compute_type = 'float16'

    model = WhisperModel(model_path, device=device, compute_type=compute_type,
                         device_index=device_index, num_workers=num_workers)

    if 'large-v3' in model_path:
        model.feature_extractor.mel_filters = \
            model.feature_extractor.get_mel_filters(model.feature_extractor.sampling_rate,
                                                    model.feature_extractor.n_fft, n_mels=128)

    dataloader = get_dataloader(dataset_dir, None, batch_size=32, shuffle=False, type='whisper_faster')

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    with TrainMonitor() as monitor:
        for batch in tqdm(dataloader):
            data_path, ref, idx = batch
            data_path = data_path[0]
            ref = ref[0]
            idx = idx[0]

            with StepCounter(handle) as ct:
                segments, info = model.transcribe(audio=data_path, language=language)
                for segment in segments:
                    transcription = segment.text

            cost_time = ct.cost_time
            memory_used = ct.cost_memory
            cpu_usage = ct.cpu_usage

            monitor.total_cost_time += cost_time
            monitor.memory.append(memory_used)
            monitor.max_cpu_usage = max(cpu_usage, monitor.max_cpu_usage)
            monitor.total_audio_time += get_duration_from_idx(idx)
            monitor.refs.append(f'{ref} ({idx})')
            monitor.trans.append(f'{transcription} ({idx})')
            monitor.trans_with_info.append(f'{transcription} ({idx})'
                                           f'batch-info: cost time: {cost_time} '
                                           f'used memory: {memory_used} '
                                           f'cpu usage: {cpu_usage}')

    export_dir += f'-{device}-{compute_type}'
    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)


def eval_whisper_huggingface(model_path: str, dataset_dir: str, export_dir: str,
                             batch_size: int, language: str, device: torch.device) -> None:
    model, processor = load_whisper(model_path)
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, processor, batch_size, shuffle=False, type='whisper_huggingface')
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    model.to(device)
    model.eval()
    with TrainMonitor() as monitor:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    input_features, ref, idx = batch
                    input_features = input_features.to(device)

                    with StepCounter(handle) as ct:
                        predicted_ids = model.generate(input_features, task='transcribe', language=language)
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                    cost_time = ct.cost_time
                    memory_used = ct.cost_memory
                    cpu_usage = ct.cpu_usage
                    print(cpu_usage)
                    monitor.total_cost_time += cost_time
                    monitor.memory.append(memory_used)
                    monitor.max_cpu_usage = max(cpu_usage, monitor.max_cpu_usage)

                    for i in range(len(transcription)):
                        monitor.total_audio_time += get_duration_from_idx(idx[i])
                        monitor.refs.append(f'{ref[i]} ({idx[i]})')
                        monitor.trans.append(f'{transcription[i]} ({idx[i]})')
                        if i == 0:
                            monitor.trans_with_info.append(f'batch-info: cost time: {cost_time} '
                                                           f'used memory: {memory_used} '
                                                           f'cpu usage: {cpu_usage}')
                            monitor.trans_with_info.append(f'{transcription[i]} ({idx[i]}) ')
                        else:
                            monitor.trans_with_info.append(f'{transcription[i]} ({idx[i]})')

    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)


def eval_whisper_pipeline(model_path: str, dataset_dir: str, export_dir: str,
                          batch_size: int, language: str, device: torch.device) -> None:
    pipe = get_pipeline(model_path, batch_size, gpu=str(device.index))
    processor = WhisperProcessor.from_pretrained(model_path)
    generate_kwargs = {"task": 'transcribe', "num_beams": 1, "language": language}
    dataloader = get_dataloader(dataset_dir, processor, batch_size, shuffle=False, type='whisper_openai')
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    with TrainMonitor as monitor:
        for batch in tqdm(dataloader):
            data_path, ref, idx = batch
            data_path = list(data_path)

            with StepCounter(handle) as ct:
                result = pipe(data_path, return_timestamps=False, generate_kwargs=generate_kwargs)

            cost_time = ct.cost_time
            memory_used = ct.cost_memory
            cpu_usage = ct.cpu_usage

            monitor.total_cost_time += cost_time
            monitor.memory.append(memory_used)
            monitor.max_cpu_usage = max(cpu_usage, monitor.max_cpu_usage)

            for i in range(len(result)):
                transcription = result[i]['text']
                monitor.total_audio_time += get_duration_from_idx(idx[i])
                monitor.refs.append(f'{ref[i]} ({idx[i]})')
                monitor.trans.append(f'{transcription} ({idx[i]})')
                if i == 0:
                    monitor.trans_with_info.append(f'batch-info: cost time: {cost_time} '
                                           f'used memory: {memory_used} '
                                           f'cpu usage: {cpu_usage}')
                    monitor.trans_with_info.append(f'{transcription} ({idx[i]}) ')
                else:
                    monitor.trans_with_info.append(f'{transcription} ({idx[i]})')

    export_dir += f'-pipeline'
    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)
