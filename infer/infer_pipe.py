import argparse
import os
import platform
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCausalLM
from utils.get_save_file import get_file
from utils.count_time import CountTime
from utils.get_audio_duration import get_duration_from_idx


def get_pipeline(args) -> pipeline:
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}')
    torch_dtype = torch.float16 if torch.cuda.is_available() and args.use_gpu else torch.float32

    # 获取Whisper的特征提取器、编码器和解码器
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 获取模型
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True,
        use_flash_attention_2=args.use_flash_attention_2
    )
    print(args.use_bettertransformer, args.use_flash_attention_2)
    if args.use_bettertransformer and not args.use_flash_attention_2:
        print('use bettertransformer')
        model = model.to_bettertransformer()

    # 使用Pytorch2.0的编译器
    if args.use_compile:
        if torch.__version__ >= "2" and platform.system().lower() != 'windows':
            model = torch.compile(model)
    model.to(device)

    # 获取助手模型
    generate_kwargs_pipeline = None
    if args.assistant_model_path is not None:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            args.assistant_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        assistant_model.to(device)
        generate_kwargs_pipeline = {"assistant_model": assistant_model}

    # 获取管道
    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=args.batch_size,
                    torch_dtype=torch_dtype,
                    generate_kwargs=generate_kwargs_pipeline,
                    device=device)
    return pipe


def infer_single(pipe: pipeline, audio_path: str, generate_kwargs=None) -> None:
    result = pipe(audio_path, return_timestamps=False, generate_kwargs=generate_kwargs)
    for chunk in result["chunks"]:
        print(f"[{chunk['timestamp'][0]}-{chunk['timestamp'][1]}s] {chunk['text']}")


def infer_list(pipe: pipeline, audio_dir: str, generate_kwargs=None) -> None:
    '''

    Args:
        pipe: transformers pipeline
        audio_dir: audio_dir/wav.scp
        generate_kwargs: {task, num_beams, language}

    Returns:

    '''
    file = get_file(os.path.join(audio_dir, 'wav.scp'))
    audio_path_list = [n.split(' ')[-1] for n in file]
    total_time = 0
    # for line in file:
    #     total_time += get_duration_from_idx(line.split(' ')[0])
    #
    # print(total_time)
    with CountTime() as ct:
        result = pipe(audio_path_list[:16], return_timestamps=False, generate_kwargs=generate_kwargs)

    cost_time = ct.cost_time
    print(cost_time)
    for item in result:
        print(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_path",  type=str,  default=None, help="单音频路径")
    parser.add_argument("--audio_dir",   type=str,  default=None, help="存放wav.scp的目录")
    parser.add_argument("--model_path",  type=str,  default="models/whisper-tiny-finetune/", help="合并模型的路径，或者是huggingface上模型的名称")
    parser.add_argument("--use_gpu",     type=bool, default=True,      help="是否使用gpu进行预测")
    parser.add_argument("--language",    type=str,  default="chinese", help="设置语言，如果为None则预测的是多语言")
    parser.add_argument("--num_beams",   type=int,  default=1,         help="解码搜索大小")
    parser.add_argument("--batch_size",  type=int,  default=16,        help="预测batch_size大小")
    parser.add_argument("--use_compile", type=bool, default=False,     help="是否使用Pytorch2.0的编译器")
    parser.add_argument("--task",        type=str,  default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
    parser.add_argument("--gpu",         type=str,  default="0",       help="gpu序号")
    parser.add_argument("--assistant_model_path",  type=str,  default=None,  help="助手模型，可以提高推理速度，例如openai/whisper-tiny")
    parser.add_argument("--local_files_only",      type=bool, default=True,  help="是否只在本地加载模型，不尝试下载")
    parser.add_argument("--use_flash_attention_2", type=bool, default=False, help="是否使用FlashAttention2加速")
    parser.add_argument("--use_bettertransformer", type=bool, default=True, help="是否使用BetterTransformer加速")
    args = parser.parse_args()

    # 得到 pipeline
    infer_pipe = get_pipeline(args)
    # 推理参数
    generate_kwargs = {"task": args.task, "num_beams": args.num_beams}
    if args.language is not None:
        generate_kwargs["language"] = args.language

    # 推理
    if args.audio_path is not None:
        infer_single(infer_pipe, args.audio_path, generate_kwargs)
    if args.audio_dir is not None:
        infer_list(infer_pipe, args.audio_dir, generate_kwargs)
    else:
        print('Need audio.')

