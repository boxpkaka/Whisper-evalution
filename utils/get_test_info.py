import os
import re
import torch
from utils.get_save_file import get_json


def get_test_info(args):
    config = get_json(args.config)
    test_info = {}
    model_type = args.model_type
    model_dir = os.path.join(config['model_dir'], model_type)
    model_name = config['model_name_list'][model_type][args.model_index]
    model_path = os.path.join(model_dir, model_name)

    test_set = config['dataset_list'][args.data_index]
    dataset_dir = os.path.join(config['dataset_dir'], test_set)

    # Create name of save directory
    export_postfix = dataset_dir.split('/')[-1]
    if dataset_dir.split('/')[-1] in ['train', 'dev', 'test']:
        export_postfix, split = dataset_dir.split('/')[-2:]
        export_postfix = f'{split}-{export_postfix}'
    export_dir = os.path.join(config['export_dir'], model_name + '-' + export_postfix)

    device = torch.device(f'cuda:{args.gpu}')

    test_info[f'model'] = f'{args.model_type} & {model_name}'
    test_info[f'test set'] = f'{test_set}'
    test_info[f'export'] = f'{export_dir}'
    test_info[f'batch size'] = f'{args.batch_size}'
    test_info[f'language'] = f'{args.language}'
    test_info[f'num_workers'] = f'{args.num_workers}'
    test_info[f'gpu'] = f'{args.gpu}'

    test_kwargs = {
        'model_path': model_path,
        'dataset_dir': dataset_dir,
        'export_dir': export_dir,
        'language': args.language,
        'num_workers': args.num_workers,
        'device': device
    }

    if re.match('openai', model_name) is not None:
        res = ['openai', test_kwargs]
    elif re.match('faster', model_name) is not None:
        test_kwargs['use_cpu'] = True if args.use_cpu else False
        test_kwargs['compute_type'] = args.compute_type
        test_info['use_cpu'] = test_kwargs['use_cpu']
        test_info['compute_type'] = test_kwargs['compute_type']
        res = ['faster', test_kwargs]
    elif args.pipeline >= 1:
        test_kwargs['use_flash_attention_2'] = True if args.use_flash_attention_2 > 0 else False
        test_kwargs['use_bettertransformer'] = True if args.use_bettertransformer > 0 else False
        test_kwargs['use_compile'] = True if args.use_compile > 0 else False
        test_kwargs['assistant_model_path'] = args.assistant_model_path
        test_info['use_flash_attention_2'] = test_kwargs['use_flash_attention_2']
        test_info['use_bettertransformer'] = test_kwargs['use_bettertransformer']
        test_info['use_compile'] = test_kwargs['use_compile']
        test_info['assistant_model_path'] = test_kwargs['assistant_model_path']
        res = ['pipeline', test_kwargs]
    else:
        test_kwargs['LoRA dir'] = args.lora_dir
        test_info['LoRA dir'] = args.lora_dir
        res = ['huggingface', test_kwargs]

    max_length = max([len(k) for k in test_info.keys()])
    for k, v in test_info.items():
        padding = max_length - len(k)
        print(f'{k} ' + padding * ' ' + f'{v}')

    return res



