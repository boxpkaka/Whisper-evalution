import torch
import pynvml
from tqdm import tqdm
from faster_whisper import WhisperModel

from utils import (
    StepCounter,
    TrainMonitor,
    get_duration_from_idx,
    get_dataloader
)
from eval.whisper_hf import save_eval


def eval_faster_whisper(
        model_path: str,
        dataset_dir: str,
        export_dir: str,
        language: str,
        use_cpu: bool,
        num_workers: int,
        compute_type: str,
        device: torch.device):

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

