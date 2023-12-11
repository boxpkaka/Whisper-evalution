import platform
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_whisper(path: str):
    whisper = WhisperForConditionalGeneration.from_pretrained(path, local_files_only=True)
    whisper_processor = WhisperProcessor.from_pretrained(path, local_files_only=True)

    return whisper, whisper_processor


def get_pipeline(model_path: str, batch_size: int, gpu: str, use_flash_attention_2=None,
                 use_bettertransformer=None, use_compile=None, assistant_model_path=None) -> pipeline:
    # 设置设备
    device = torch.device(f'cuda:{gpu}')
    torch_dtype = torch.float16

    # 获取Whisper的特征提取器、编码器和解码器
    processor = AutoProcessor.from_pretrained(model_path)

    # 获取模型
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True,
        use_flash_attention_2=use_flash_attention_2
    )

    if use_bettertransformer and not use_flash_attention_2:
        model = model.to_bettertransformer()

    # 使用Pytorch2.0的编译器
    if use_compile:
        if torch.__version__ >= "2" and platform.system().lower() != 'windows':
            model = torch.compile(model)
    model.to(device)

    # 获取助手模型
    generate_kwargs_pipeline = None
    if assistant_model_path is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(device)
        print('assistant model has been loaded')
        generate_kwargs_pipeline = {"assistant_model": assistant_model}

    # 获取管道
    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=2,
                    batch_size=batch_size,
                    torch_dtype=torch_dtype,
                    generate_kwargs=generate_kwargs_pipeline,
                    device=device)
    return pipe

