import argparse
import os
import soundfile
import librosa
from transformers import pipeline
from utils.get_save_file import get_file
from utils.count_time import CountTime
from utils.get_model import get_pipeline


def infer_single(pipe: pipeline, audio_path: str, generate_kwargs=None) -> None:
    wav, sr = soundfile.read(audio_path)
    if sr != 16000:
        print('resample')
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    result = pipe(wav, return_timestamps=False, generate_kwargs=generate_kwargs)
    print(result)


def infer_list(pipe: pipeline, audio_dir: str, generate_kwargs=None) -> None:
    file = get_file(os.path.join(audio_dir, 'wav.scp'))
    audio_path_list = [n.split(' ')[-1] for n in file]
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
    parser.add_argument("--use_bettertransformer", type=bool, default=False, help="是否使用BetterTransformer加速")
    args = parser.parse_args()

    # 得到 pipeline
    infer_pipe = get_pipeline(model_path=args.model_path,
                              batch_size=args.batch_size,
                              gpu=args.gpu,
                              use_bettertransformer=args.use_bettertransformer,
                              use_flash_attention_2=args.use_flash_attention_2,
                              use_compile=args.use_compile,
                              assistant_model_path=args.assistant_model_path
                              )
    # 推理参数
    generate_kwargs = {"task": args.task, "num_beams": args.num_beams}
    if args.language is not None:
        generate_kwargs["language"] = args.language

    # 推理
    if args.audio_path is not None:
        infer_single(infer_pipe, args.audio_path, generate_kwargs)
    elif args.audio_dir is not None:
        infer_list(infer_pipe, args.audio_dir, generate_kwargs)
    else:
        print('Need audio.')

