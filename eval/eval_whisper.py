import torch
import whisper
import os
import pynvml
from tqdm import tqdm
from faster_whisper import WhisperModel
from peft import PeftConfig, inject_adapter_in_model

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


def eval_whisper_openai(model_path: str, dataset_dir: str, export_dir: str, language: str,
                        num_workers: int, device: torch.device):
    dataloader = get_dataloader(dataset_dir, batch_size=1, num_workers=num_workers, shuffle=False, return_type='path')
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


def eval_faster_whisper(model_path: str, dataset_dir: str, export_dir: str, language: str, use_cpu: bool,
                        num_workers: int, compute_type: str, device: torch.device):
    print('=' * 100)
    dataloader = get_dataloader(dataset_dir, batch_size=1, shuffle=False, num_workers=num_workers, return_type='dict')
    device_index = device.index
    if use_cpu:
        kwargs = {'model_size_or_path': model_path, 'device': 'cpu', 'num_workers': num_workers, 'cpu_threads': 32}
    else:
        kwargs = {'model_size_or_path': model_path, 'device': 'cuda', 'compute_type': compute_type,
                 'device_index': device_index, 'num_workers': num_workers}

    model = WhisperModel(**kwargs)

    if 'large-v3' in model_path:
        model.feature_extractor.mel_filters = \
            model.feature_extractor.get_mel_filters(model.feature_extractor.sampling_rate,
                                                    model.feature_extractor.n_fft, n_mels=128)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    with TrainMonitor() as monitor:
        for batch in tqdm(dataloader):
            data, ref, idx = batch
            data = data[0]['raw']
            ref = ref[0]
            idx = idx[0]

            with StepCounter(handle) as ct:
                segments, info = model.transcribe(audio=data, language=language)
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


def eval_whisper_huggingface(model_path: str, dataset_dir: str, export_dir: str, batch_size: int,
                             language: str, num_workers: int, device: torch.device, lora_dir=None) -> None:
    model, processor = load_whisper(model_path)
    if lora_dir is not None:
        peft_config = PeftConfig.from_pretrained(lora_dir)
        model = inject_adapter_in_model(peft_config, model)
        print('Lora has been loaded!')
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, batch_size, shuffle=False, num_workers=num_workers,
                                return_type='feature', processor=processor)
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    model.to(device)
    model.eval()
    with TrainMonitor() as monitor:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_features, ref, idx = batch
                input_features = input_features.to(device)

                with StepCounter(handle) as ct:
                    with torch.cuda.amp.autocast(enabled=True):
                        predicted_ids = model.generate(input_features, task='transcribe', language=language)
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                cost_time = ct.cost_time
                print(cost_time)
                memory_used = ct.cost_memory
                cpu_usage = ct.cpu_usage

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
    if lora_dir is not None:
        export_dir += lora_dir.split('/')[-1]
    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)


def eval_whisper_pipeline(model_path: str, dataset_dir: str, export_dir: str,
                          batch_size: int, language: str, device: torch.device) -> None:
    pipe = get_pipeline(model_path, batch_size, gpu=str(device.index), assistant_model_path='/data1/yumingdong/model/huggingface/whisper-small')
    generate_kwargs = {"task": 'transcribe', "num_beams": 1, "language": language}
    dataloader = get_dataloader(dataset_dir, batch_size, shuffle=False, type='dict')
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    data_set = []
    ref_set = []
    idx_set = []
    for batch in tqdm(dataloader):
        data, ref, idx = batch
        data_set.extend(data)
        ref_set.extend(ref)
        idx_set.extend(idx)

    with TrainMonitor() as monitor:
        with StepCounter(handle) as ct:
            result = pipe(data_set, return_timestamps=False, generate_kwargs=generate_kwargs)

        cost_time = ct.cost_time
        memory_used = ct.cost_memory
        cpu_usage = ct.cpu_usage

        monitor.total_cost_time += cost_time
        monitor.memory.append(memory_used)
        monitor.max_cpu_usage = max(cpu_usage, monitor.max_cpu_usage)

        for i in range(len(result)):
            transcription = result[i]['text']
            monitor.total_audio_time += get_duration_from_idx(idx_set[i])
            monitor.refs.append(f'{ref_set[i]} ({idx_set[i]})')
            monitor.trans.append(f'{transcription} ({idx_set[i]})')
            if i == 0:
                monitor.trans_with_info.append(f'batch-info: cost time: {cost_time} '
                                               f'used memory: {memory_used} '
                                               f'cpu usage: {cpu_usage}')
                monitor.trans_with_info.append(f'{transcription} ({idx_set[i]}) ')
            else:
                monitor.trans_with_info.append(f'{transcription} ({idx_set[i]})')

    export_dir += f'-pipeline'
    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)
