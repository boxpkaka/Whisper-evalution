import os
import torch
import pynvml
import whisper
from tqdm import tqdm

from utils import (
    count_model,
    StepCounter,
    TrainMonitor,
    get_duration_from_idx,
    get_dataloader
)
from eval.whisper_hf import save_eval
from model.get_model import load_hf_processor


def eval_whisper_openai(
        model_path: str,
        dataset_dir: str,
        export_dir: str,
        batch_size: int,
        language: str,
        num_workers: int,
        device: torch.device):

    model = whisper.load_model(os.path.join(model_path, 'model.pt'), device=device)
    processor = load_hf_processor('/data1/yumingdong/model/huggingface/whisper-large-v3')
    dataloader = get_dataloader(dataset_dir, batch_size=1, num_workers=num_workers,
                                shuffle=False, return_type='path', processor=processor)
    param = count_model(model)
    print(param)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    with TrainMonitor() as monitor:
        for batch in tqdm(dataloader):
            path, ref, idx = batch
            path = path[0]
            ref = ref[0]
            idx = idx[0]

            with StepCounter(handle) as ct:
                transcription = model.transcribe(path, language=language)['text']
                print(transcription)
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

    export_dir += 'openai'
    save_eval(export_dir, monitor.refs, monitor.trans, monitor.trans_with_info)

