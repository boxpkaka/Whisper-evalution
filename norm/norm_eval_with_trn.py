from eval_with_trn import eval_with_trn
from norm_with_trn import normalize_cantonese
import os

path = '/data1/yumingdong/whisper/whisper-eval/exp/whisper-large-v3-lora50-14000-test_aishell'

normalize_cantonese(path)
# eval_with_trn(path)
