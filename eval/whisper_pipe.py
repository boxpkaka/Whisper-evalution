import torch
import pynvml
from tqdm import tqdm

from utils import (
    StepCounter,
    TrainMonitor,
    get_duration_from_idx,
    get_dataloader
)
from model import get_pipeline
from eval.whisper_hf import save_eval


def eval_whisper_pipeline(
                model_path: str,
                dataset_dir: str,
                export_dir: str,
                batch_size: int,
                num_workers: int,
                language: str,
                device: torch.device,
                use_flash_attention_2=None,
                use_bettertransformer=None,
                use_compile=None,
                assistant_model_path=None,
                torch_dtype=torch.float32,
                ) -> None:

    pipe = get_pipeline(model_path,
                        batch_size,
                        gpu=str(device.index),
                        use_flash_attention_2=use_flash_attention_2,
                        use_bettertransformer=use_bettertransformer,
                        use_compile=use_compile,
                        assistant_model_path=assistant_model_path,
                        torch_dtype=torch_dtype)
    generate_kwargs = {"task": 'transcribe', "num_beams": 1, "language": language}
    dataloader = get_dataloader(dataset_dir, 64, shuffle=False, num_workers=num_workers, return_type='dict')
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

