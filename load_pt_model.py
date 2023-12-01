import torch
import torch.nn as nn
# from wenet.utils.init_model import init_model
# from funasr.build_utils.build_model import build_model
import argparse
import yaml
import json

def load_config(yaml_path, cmvn_file=None):
  """加载配置文件
    args:
      pt_path: 配置文件路径
      cmvn_file: cmvn文件路径

    return:
      参数字典数据
  """

  print(f"load config.yaml {yaml_path}")
  with open(yaml_path, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
  # configs['input_dim'] = configs['dataset_conf']['fbank_conf']['num_mel_bins']
  if cmvn_file:
    configs['cmvn_file'] = cmvn_file
  return configs


def load_args(yaml_path):
  """加载配置文件
    args:
      pt_path: 配置文件路径
      cmvn_file: cmvn文件路径

    return:
      参数字典数据
  """
  parser = argparse.ArgumentParser()
  configs = load_config(yaml_path)
  args = parser.parse_args()
  return configs


def load_model(pt_path, config=None):
  """加载pt模型文件
    args:
      pt_path: pt模型路径

    return:
      模型结构
  """
  if config:
    # model = init_model(config) # wenet
    print(config.keys())
    model = build_model(config)
    checkpoint = torch.load(pt_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    
  else:
    model = torch.load(pt_path, map_location='cpu')
  return model


def print_model(model):
  """打印模型文件
    args:
      model: collections.OrderedDict 有序的字典结构
  """
  for key, value in model.items():
    print(key, type(value))


def _compare_model(model1, model2, changed=False):
  """比较两个模型的每一层参数是否有变化, 打印变换的层
    args:
      model1: 第一个模型参数;
      model2: 第二个模型参数;
      torch.equal(value, model2.state_dict()[key]):
  """
  for key, value in model1.items():
    if (not torch.equal(value, model2[key])) and changed:
      print(f"Parameter {key} has changed.")
    else:
      print(f"Parameter {key} is same.")


def compare_model(model1_path, model2_path, config_path, cmvn_file):
  if config_path is not None:
    config = load_config(config_path)
  else:
    config = None
  model1 = load_model(model1_path, config)
  print_model(model1)
  # model2 = load_model(model2_path, config)
  # _compare_model(model1, model2)

if __name__ == "__main__":
  root_file = "/data1/cuidc/20220425/FunASR/egs_modelscope/asr/paraformer/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
  cmvn_file = ""
  config_file = root_file + "/checkpoint-large-lr5.0e-04/config.yaml"
  model1_file = root_file + "/checkpoint-large-lr5.0e-04/11epoch.pb"
  model2_file = root_file + "/checkpoint-large-lr5.0e-04/model.pb"

  model_file = "/data1/cuidc/.cache/modelscope/hub/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.pb"
  compare_model(model1_file, model2_file, None, cmvn_file)
