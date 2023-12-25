import torch
import os
import pynvml
from tqdm import tqdm
from peft import PeftConfig, inject_adapter_in_model

from .eval_with_trn import eval_with_trn
from norm import normalize_cantonese
from utils import (
    count_model,
    save_file,
    StepCounter,
    TrainMonitor,
    get_duration_from_idx,
    get_dataloader
)
from model.get_model import load_hf_whisper, load_hf_processor


def save_eval(export_dir, refs, trans, trans_with_time=None):
    os.makedirs(export_dir, exist_ok=True)
    save_file(os.path.join(export_dir, 'std_orig.trn'), refs)
    save_file(os.path.join(export_dir, 'reg_orig.trn'), trans)
    if trans_with_time is not None:
        save_file(os.path.join(export_dir, 'reg_rtf.trn'), trans_with_time)
    normalize_cantonese(export_dir)
    eval_with_trn(export_dir)


def eval_whisper_huggingface(
        model_path: str,
        dataset_dir: str,
        export_dir: str,
        batch_size: int,
        language: str,
        num_workers: int,
        device: torch.device,
        lora_dir=None,
        use_flash_attention_2=False,
        torch_dtype=torch.float32) -> None:

    model = load_hf_whisper(model_path, use_flash_attention_2, torch_dtype)
    processor = load_hf_processor(model_path)

    if lora_dir is not None:
        peft_config = PeftConfig.from_pretrained(lora_dir)
        model = inject_adapter_in_model(peft_config, model)
        print('LoRA has been loaded!')
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, batch_size, shuffle=False, num_workers=num_workers,
                                return_type='feature', processor=processor)
    print('=' * 100)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    model.to(device)

    preheat = True
    if preheat is True:
        print('Start preheat...')
        for _ in tqdm(range(3)):
            for batch in tqdm(dataloader):
                input_features = batch[0].to(device).to(torch_dtype)
                with torch.cuda.amp.autocast(enabled=True):
                    _ = model.generate(input_features, task='transcribe', language=language)
                break

    print('Start eval...')
    with TrainMonitor() as monitor:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_features, ref, idx = batch
                input_features = input_features.to(device).to(torch_dtype)
                generate_fn = model.generate
                with StepCounter(handle) as ct:
                    with torch.cuda.amp.autocast(enabled=True):
                        predicted_ids = generate_fn(input_features, task='transcribe', language=language)
                        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

                cost_time = ct.cost_time
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

