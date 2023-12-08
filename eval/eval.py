import re
import os
import argparse
from eval.eval_whisper import *
from eval.eval_mms import *
from utils.get_save_file import get_json


def eval_whisper(args):
    config = get_json(args.config)
    print(config)
    model_dir = os.path.join(config['model_dir'], args.model_type)

    model_name = config['model_name_list'][args.model_type][args.model_index]
    model_path = os.path.join(model_dir, model_name)

    dataset_dir = os.path.join(config['dataset_dir'], config['dataset_list'][args.data_index])

    # Create name of save directory
    export_postfix = dataset_dir.split('/')[-1]
    if dataset_dir.split('/')[-1] in ['train', 'dev', 'test']:
        export_postfix, split = dataset_dir.split('/')[-2:]
        export_postfix = f'{split}-{export_postfix}'

    export_dir = os.path.join(config['export_dir'], model_name + '-' + export_postfix)

    device = torch.device(f'cuda:{args.gpu}')

    print('=' * 100)
    print('model:        ', f'{args.model_type} & {model_name}')
    print('language:     ', args.language)
    print('test set:     ', config['dataset_list'][args.data_index])
    print('export:       ', export_dir)
    print('gpu:          ', args.gpu)
    print('batch size:   ', args.batch_size)
    print('compute type: ', args.compute_type)

    if re.match('openai', model_name) is not None:
        eval_whisper_openai(model_path, dataset_dir, export_dir, args.language, device)
    elif re.match('faster', model_name) is not None:
        eval_faster_whisper(model_path, dataset_dir, export_dir, args.language,
                            args.use_cpu, args.num_workers, args.compute_type, device)
    elif re.match('m-ctc', model_name) is not None:
        eval_mms(model_path, dataset_dir, export_dir, device)
    elif args.pipeline:
        eval_whisper_pipeline(model_path, dataset_dir, export_dir, args.batch_size, args.language, device)
    else:
        eval_whisper_huggingface(model_path, dataset_dir, export_dir, args.batch_size, args.language,
                                 args.num_workers, device, args.lora_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--config',                 help='config path'  ,              type=str)
    parser.add_argument('--model_type',             help='type of model',              type=str)
    parser.add_argument('--model_index',            help='index of model list',        type=int)
    parser.add_argument('--lora_dir', default=None, help='directory of LoRA file',     type=str)
    parser.add_argument('--data_index',             help='index of dataset list',      type=int)
    parser.add_argument('--language',               help='whisper inference language', type=str)
    parser.add_argument('--batch_size',             help='batch size',                 type=int)
    parser.add_argument('--use_cpu', default=False, help='use cpu of ct2model',        type=bool)
    parser.add_argument('--compute_type',           help='auto/int8/int8_float16...',  type=str)
    parser.add_argument('--num_workers',            help='workers nums of ct2model',   type=int)
    parser.add_argument('--pipeline',               help='use transformers pipeline',  type=bool)
    parser.add_argument('--gpu',     default=0,     help='gpu id',                     type=str)
    args = parser.parse_args()

    eval_whisper(args)
