import os
import re
import torch
import argparse
from eval_whisper import (eval_mms,
                          eval_whisper_huggingface,
                          eval_faster_whisper,
                          eval_whisper_openai)


model_name_list = ['whisper-large-v3',
                   'whisper-large-v3-lora500-final',
                   'whisper-large-v3-lora50-14000',
                   'whisper-large-v3-lora50-14000-attn-yue',
                   'whisper-large-v3-lora50-14000-attn-none',
                   'whisper-large-v3-lora50+50-14000-attn-none']

dataset_list = ['test_1000Cantonese',
                'test_datatang500h',
                'test_magicdatacantonese',
                'test_commonvoicecantonese',
                'test_kejiyuan',
                'test_hk_can',
                'dev_mandarin_2h',
                'aishell/test']


def eval_whisper(model_dir: str, model_type: str, model_index: int,
                 dataset_dir: str, data_index: int,
                 export_dir: str,
                 language: str, batch_size: int, gpu: str):
    model_dir = os.path.join(model_dir, model_type)

    model_name = model_name_list[model_index]
    model_path = os.path.join(model_dir, model_name)

    dataset_dir = os.path.join(dataset_dir, dataset_list[data_index])

    export_postfix = dataset_dir.split('/')[-1] if data_index != 6 else 'test_aishell'
    export_dir = os.path.join(export_dir, model_name + '-' + export_postfix)

    device = torch.device(f'cuda:{gpu}')

    print('=' * 100)
    print('model:    ', model_name)
    print('language: ', language)
    print('test set: ', dataset_list[data_index])
    print('export:   ', export_dir)
    print('gpu:      ', gpu)

    if re.match('openai', model_name) is not None:
        eval_whisper_openai(model_path, dataset_dir, export_dir, language, device)
    elif re.match('faster', model_name) is not None:
        eval_faster_whisper(model_path, dataset_dir, export_dir, language, device)
    elif re.match('m-ctc', model_name) is not None:
        eval_mms(model_path, dataset_dir, export_dir, device)
    else:
        eval_whisper_huggingface(model_path, dataset_dir, export_dir, batch_size, language, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--model_dir', '-md', help='model root directory', type=str)
    parser.add_argument('--model_type', '-mt', help='type of model',
                        choices=['huggingface', 'finetuned', 'openai_whisper', 'wenet_whisper'],
                        type=str)
    parser.add_argument('--model_index', '-mi', help='index of model list', type=int)
    parser.add_argument('--dataset_dir', '-dd', help='dataset root directory', type=str)
    parser.add_argument('--data_index', '-di', help='index of dataset list', type=int)
    parser.add_argument('--export_dir', '-ed', help='export directory of result', type=int)
    parser.add_argument('--language', '-l', help='whisper inference language', type=str)
    parser.add_argument('--batch_size', '-b', help='batch size', type=int)
    parser.add_argument('--gpu', '-g', default=0, help='gpu id', type=str)
    args = parser.parse_args()

    eval_whisper(model_dir=args.model_dir,
                 model_type=args.model_type,
                 model_index=args.model_index,
                 dataset_dir=args.dataset_dir,
                 data_index=args.data_index,
                 export_dir=args.export_dir,
                 language=args.language,
                 batch_size=args.batch_size,
                 gpu=args.gpu)
