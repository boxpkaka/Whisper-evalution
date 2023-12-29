import torch
from model import get_pipeline


def infer_list(model_path: str, audio_path: list, language: str, device: torch.device) -> None:
    gpu = str(device.index)
    pipe = get_pipeline(
        model_path=model_path,
        batch_size=16,
        gpu=gpu,
        use_bettertransformer=False,
        use_flash_attention_2=True,
        use_compile=False,
        assistant_model_path=None
        )

    generate_kwargs = {"task": 'transcribe', "num_beams": 10, "language": language}

    result = pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
    for item in result:
        print(item)

