import argparse
from train.trainer import *
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch', '-b', help='batch size')
    parser.add_argument('--modelindex', '-m', help='index of model')
    parser.add_argument('--gpu', '-g', default=0, help='gpu id')
    parser.add_argument('--usehalf', '-uh', default=0, help='use half precision, 1:yes')
    parser.add_argument('--useamp', '-ua', default=0, help='use amp, 1:yes')

    args = parser.parse_args()

    batch_size = int(args.batch)
    model_index = int(args.modelindex)
    gpu_id = args.gpu
    half = True if args.usehalf == '1' else False
    amp = True if args.useamp == '1' else False

    root_dir = '/data1/yumingdong/pretrain_model'
    model_name = ['hubert-base-ls960',
                  'hubert-large-ls960-ft',
                  'distil-large-v2',
                  'distil-medium.en',
                  'whisper-small',
                  'whisper-medium',
                  'whisper-large-v2',
                  'whisper-large-v3',
                  'faster-whisper-large-v2',
                  'faster-whisper-medium']
    model_path = [os.path.join(root_dir, n) for n in model_name]
    audio_path = '/data1/yumingdong/test/data_ls/train-clean-100'
    device = torch.device("cuda:" + gpu_id)

    print('batch size:     ', batch_size)
    print('model:          ', model_name[model_index])
    print('gpu id:         ', gpu_id)
    print('half precision: ', half)
    print('use amp:        ', amp)

    if model_index >= 2:
        model = Whisper(model_path[model_index]).to(device)
    else:
        model = HuBERT(model_path[model_index]).to(device)

    if amp:
        train_with_amp(model, batch_size, device, half)
    else:
        train(model, batch_size, device, half)




