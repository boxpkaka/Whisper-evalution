from base_utils import evaluation
from pathlib import Path
from base_utils.evaluation import SplitType
from base_utils.evaluation import write_trn_trx_list
import sys


def eval_with_trn(path):
    out_dir = Path(path)
    num_most_wrong = 1000
    std_path = Path(out_dir / Path("std.trn"))
    #std_char_path = Path("/data1/fangcheng1050/workspace/speech_corpus_construction_v1/speech_corpus_construction/train_bpojd_asr_3.7.0_3/std_char.trn")
    reg_path = Path(out_dir / Path("reg.trn"))
    #reg_char_path = Path("/data1/fangcheng1050/workspace/speech_corpus_construction_v1/speech_corpus_construction/train_bpojd_asr_3.7.0_3/reg_char.trn")

    eval_obj = evaluation.CnAsrEvalDir(out_dir, language='yue')
    std_map = evaluation.get_trn_map(std_path, SplitType.WORD)
    reg_map = evaluation.get_trn_map(reg_path, SplitType.WORD)
    eval_obj.from_trn_map(std_map, reg_map, num_most_wrong)
    # std_char_map = evaluation.get_trn_map(std_char_path, SplitType.WORD)
    # reg_char_map = evaluation.get_trn_map(reg_char_path, SplitType.WORD)
    # eval_obj.from_char_trn_map(std_char_map, reg_char_map, num_most_wrong)

    # write_trn_trx_list(std_map.items(), std_path)
    # write_trn_trx_list(reg_map.items(), reg_path)
    # eval_obj._gen_dtl_most_wrong(std_path, reg_path, SplitType.WORD, 100)