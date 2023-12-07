import re
import os
import argparse
from eval.eval_whisper import *
from eval.eval_mms import *
from eval.model_data_list import model_name_list, dataset_list


def eval_whisper(args):
    model_dir = os.path.join(args.model_dir, args.model_type)

    model_name = model_name_list[args.model_index]
    model_path = os.path.join(model_dir, model_name)

    dataset_dir = os.path.join(args.dataset_dir, dataset_list[args.data_index])

    # Create name of save directory
    export_postfix = dataset_dir.split('/')[-1]
    if dataset_dir.split('/')[-1] in ['train', 'dev', 'test']:
        export_postfix, split = dataset_dir.split('/')[-2:]
        export_postfix = f'{split}-{export_postfix}'

    export_dir = os.path.join(args.export_dir, model_name + '-' + export_postfix)

    device = torch.device(f'cuda:{args.gpu}')

    print('=' * 100)
    print('model:      ', model_name)
    print('language:   ', args.language)
    print('test set:   ', dataset_list[args.data_index])
    print('export:     ', export_dir)
    print('gpu:        ', args.gpu)
    print('batch size: ', args.batch_size)
    print('use int8:   ', args.int8)

    if re.match('openai', model_name) is not None:
        eval_whisper_openai(model_path, dataset_dir, export_dir, args.language, device)
    elif re.match('faster', model_name) is not None:
        eval_faster_whisper(model_path, dataset_dir, export_dir, args.language,
                            args.use_cpu, args.int8, args.num_workers, device)
    elif re.match('m-ctc', model_name) is not None:
        eval_mms(model_path, dataset_dir, export_dir, device)
    elif args.pipeline:
        eval_whisper_pipeline(model_path, dataset_dir, export_dir, args.batch_size, args.language, device)
    else:
        eval_whisper_huggingface(model_path, dataset_dir, export_dir, args.batch_size, args.language, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--model_dir',              help='model root directory',       type=str)
    parser.add_argument('--model_type',             help='type of model',              type=str)
    parser.add_argument('--model_index',            help='index of model list',        type=int)
    parser.add_argument('--dataset_dir',            help='dataset root directory',     type=str)
    parser.add_argument('--data_index',             help='index of dataset list',      type=int)
    parser.add_argument('--export_dir',             help='export directory of result', type=str)
    parser.add_argument('--language',               help='whisper inference language', type=str)
    parser.add_argument('--batch_size',             help='batch size',                 type=int)
    parser.add_argument('--use_cpu', default=False, help='use cpu of ct2model',        type=bool)
    parser.add_argument('--int8',                   help='ues int8 of ct2model',       type=bool)
    parser.add_argument('--num_workers',            help='workers nums of ct2model',   type=int)
    parser.add_argument('--pipeline',               help='use transformers pipeline',  type=bool)
    parser.add_argument('--gpu',     default=0,     help='gpu id',                     type=str)
    args = parser.parse_args()

    eval_whisper(args)
