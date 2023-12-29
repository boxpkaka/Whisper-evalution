import platform
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM


def load_hf_whisper(path: str, use_flash_attention_2=False, torch_dtype=torch.float32):
    flash_attn = True if use_flash_attention_2 else False
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(path,
                                                        local_files_only=True,
                                                        use_safetensors=True,
                                                        use_flash_attention_2=flash_attn,
                                                        torch_dtype=torch_dtype)
    return whisper


def load_hf_processor(path: str):
    whisper_processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    return whisper_processor


def get_pipeline(model_path: str,
                 batch_size: int,
                 gpu: str,
                 use_flash_attention_2: bool,
                 use_bettertransformer=None,
                 use_compile=None,
                 assistant_model_path=None,
                 torch_dtype=torch.float32) -> pipeline:

    device = torch.device(f'cuda:{gpu}')

    processor = load_hf_processor(model_path)
    model = load_hf_whisper(path=model_path,
                            use_flash_attention_2=use_flash_attention_2,
                            torch_dtype=torch_dtype)

    if use_bettertransformer and not use_flash_attention_2:
        model = model.to_bettertransformer()

    if use_compile:
        if torch.__version__ >= "2" and platform.system().lower() != 'windows':
            model = torch.compile(model)

    generate_kwargs_pipeline = None
    if assistant_model_path is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(device)
        print('assistant model has been loaded')
        generate_kwargs_pipeline = {"assistant_model": assistant_model}

    model.to(device)
    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=batch_size,
                    torch_dtype=torch_dtype,
                    generate_kwargs=generate_kwargs_pipeline,
                    device=device)
    return pipe

