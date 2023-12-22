#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Kuang Ru on 2018/3/2
"""提供ASR评估的相关功能."""
import re
from abc import ABCMeta, abstractmethod
from enum import Enum, unique

from base_utils.evaluation.base import AsrMetric
from base_utils.pronunciation import PronunciationDict
from base_utils.utils import get_lines_from_file
from base_utils.utils import CANTONESE_DICT_PATH, DICT_PATH

__WER_PATTERN = re.compile("= (.+)%")


@unique
class SplitType(Enum):
  """切分类型的枚举.

  Attributes:
    Split.NO: 不进行任何切分, 返回没有切分符号的句子.
    Split.CHAR: 将句子切分成字, 英文单词当作一个字.
    Split.WORD: 将句子保持原始切分.
  """
  NO = 0
  CHAR = 1
  WORD = 2


class AsrMetricPerUtt:
  """记录每一句话asr指标的类.

  Attributes:
    __std_trn_map: trn_id到标准的以字为单位的文本的映射.
    __reg_trn_map: trn_id到识别的以字为单位的文本的映射.
    __trn_id_asr_metrics: trn_id和asr指标组成的列表, 按照错误字数从高到低排序.
  """

  def __init__(self, std_trn_path, reg_trn_path):
    """初始化.

    Args:
      std_trn_path: 标准的以字为单位的trn结果文件路径.
      reg_trn_path: 识别的以字为单位的trn结果文件路径.
    """
    self.__std_trn_map = get_trn_map(std_trn_path, SplitType.WORD)
    self.__reg_trn_map = get_trn_map(reg_trn_path, SplitType.WORD)
    trn_ids = set(self.__std_trn_map.keys()) | set(self.__reg_trn_map.keys())
    self.__expand_empty(trn_ids, self.__std_trn_map)
    self.__expand_empty(trn_ids, self.__reg_trn_map)
    self.__trn_id_asr_metrics = self.__get_trn_id_asr_metrics()

  @staticmethod
  def __expand_empty(trn_ids, trn_map):
    """遍历trn_id列表, 如果trn_map内没有该trn_id, 则添加trn_id并将文本置空.

    Args:
      trn_ids: trn_id列表.
      trn_map: trn_id到文本的映射.
    """
    for trn_id in trn_ids:
      if trn_id not in trn_map:
        trn_map[trn_id] = ""

  @staticmethod
  def __get_one_trn_map(trn_id, trn_map):
    """给定一个trn_id, 得到该trn_id到文本的映射.

    Args:
      trn_id: trn_id.
      trn_map: trn_id到文本的映射.

    Returns:
      所需的trn_id到文本的映射.
    """
    trx = trn_map[trn_id] if trn_id in trn_map else ""
    return {trn_id: trx}

  def __get_trn_id_asr_metrics(self):
    """获取trn_id和asr指标组成的列表, 按照错误字数从高到低排序.

    Returns:
      trn_id和asr指标组成的列表.
    """
    trn_id_asr_metrics = list()
    for trn_id in self.__std_trn_map:
      std_trx = self.__std_trn_map[trn_id]
      reg_trx = self.__reg_trn_map[trn_id]
      asr_metric = AsrMetric({trn_id: std_trx}, {trn_id: reg_trx})
      trn_id_asr_metrics.append((trn_id, asr_metric))

    trn_id_asr_metrics.sort(key=lambda x: x[1].error_num, reverse=True)
    return trn_id_asr_metrics

  def gen_most_wrong(self, out_path, num=20):
    """生成错误字数最多的句子.

    Args:
      out_path: 输出文件路径.
      num: 数量, 默认20.
    """
    with out_path.open("w", encoding="utf-8") as out_file:
      for trn_id, met in self.__trn_id_asr_metrics[:num]:
        std_trx = self.__std_trn_map[trn_id]
        out_file.write(f"trn_id: {trn_id}\n")
        out_file.write(f"标注结果: {std_trx}\n")
        out_file.write(f"识别结果: {self.__reg_trn_map[trn_id]}\n")

        if std_trx:
          out_file.write(f"错误总计: {met.error_num}({met.error_percent:.3%})\n")
          out_file.write(f"插入错误: {met.insert_num}({met.insert_percent:.3%})\n")
          out_file.write(f"删除错误: {met.delete_num}({met.delete_percent:.3%})\n")
          out_file.write(f"替换错误: "
                         f"{met.replace_num}({met.replace_percent:.3%})\n\n")
        else:
          out_file.write(f"错误总计: {met.error_num}(NaN)\n")
          out_file.write(f"插入错误: {met.insert_num}(NaN)\n")
          out_file.write(f"删除错误: {met.delete_num}(NaN)\n")
          out_file.write(f"替换错误: {met.replace_num}(NaN)\n\n")


class AsrEvalDirBase(metaclass=ABCMeta):
  """Asr评测结果文件夹基类.

  Attributes内给定了标准评测结构文件夹的结构, 一些public方法用于生成评测结果文件夹.

  需要继承使用的方法:
  from_utts: 生成标准的评测结果文件夹, 标准结果从utterance列表获取,
             识别结果从传入的decode trn文件获取.
  from_trn_map: 生成标准的评测结果文件夹,
                标准结果和识别结果均从trn id到以字为单位的文本的映射获取.

  Attributes:
    _std_trn_path: 标准结果以词为单位的trn id到文本的映射.
    _reg_trn_path: 识别结果以词为单位的trn id到文本的映射.
    _dtl_path: 评测结果文件路径, 包含错误率等信息.
    __most_wrong: 文件内包含错误字数最多的句子, 按照错误字数从高到低排序.
    __err_statistics_path: 评测结果文件路径, 包含某些指定词的准确率和召回率等信息.
  """

  def __init__(self, eval_dir):
    """初始化.

    Args:
      eval_dir: Asr评测结果文件夹类.
    """
    eval_dir.mkdir(exist_ok=True, parents=True)
    self._std_trn_path = eval_dir / "std.trn"
    self._reg_trn_path = eval_dir / "reg.trn"
    self._dtl_path = eval_dir / "dtl"
    self.__most_wrong = eval_dir / "most_wrong"
    self.__err_statistics_path = eval_dir / "err_statistics"

  @staticmethod
  def __error_count_text(error_type, count):
    """向文件中写入某种错误类型的错误数量.

    Args:
      error_type: 错误类型.
      count: 错误数量.

    Returns:
      需要写入的错误数量的文本.
    """
    return "{:<32s} {:<21s} ({})\n\n".format(error_type, "TOTAL", count)

  def _gen_dtl_most_wrong(self, std_trn_path, reg_trn_path, split_type,
                          num_most_wrong, word_phases=None):
    """生成dtl评测结果文件和most wrong文件.

    Args:
      std_trn_path: 标签结果文件.
      reg_trn_path: 识别结果文件.
      split_type: 分词粒度, SplitType类.
      num_most_wrong: most wrong的句子数量.
      word_phases: [[词1, 词2,...],...], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    std_trn_map = get_trn_map(std_trn_path, split_type)
    reg_trn_map = get_trn_map(reg_trn_path, split_type)
    result = AsrMetric(std_trn_map, reg_trn_map, check_word_phases=word_phases)

    if word_phases:
      result.word_phase_err_and_corr(word_phases, self.__err_statistics_path)

    with self._dtl_path.open("w", encoding="utf-8") as dtl_file:
      dtl_file.write("WORD RECOGNITION PERFORMANCE\n\n")

      dtl_file.write(f"Percent Total Error       =   {result.error_percent:.3%}"
                     f"   ({result.error_num})\n\n")

      dtl_file.write(
          f"Percent Correct           =   {result.correct_percent:.3%}"
          f"   ({result.correct_num})\n\n")

      dtl_file.write(
          f"Percent Substitutions     =   {result.replace_percent:.3%}"
          f"   ({result.replace_num})\n")

      dtl_file.write(
          f"Percent Deletions         =   {result.delete_percent:.3%}"
          f"   ({result.delete_num})\n")

      dtl_file.write(
          f"Percent Insertions        =   {result.insert_percent:.3%}"
          f"   ({result.insert_num})\n")

      dtl_file.write(
          f"Percent Word Accuracy     =   {result.accuracy:.3%}\n\n\n")

      dtl_file.write(self.__error_count_text("CONFUSION PAIRS",
                                             len(result.replace)))

      for index, (src_dest, count) in enumerate(result.replace, 1):
        dtl_file.write(f"{index:4d}:{count:5d}  ->  {src_dest[0]} ==> "
                       f"{src_dest[1]}\n")

      dtl_file.write("\n\n")
      dtl_file.write(self.__error_count_text("INSERTIONS", len(result.insert)))

      for index, (char, count) in enumerate(result.insert, 1):
        dtl_file.write(f"{index:4d}:{count:5d}  ->  {char}\n")

      dtl_file.write(self.__error_count_text("DELETIONS", len(result.delete)))

      for index, (char, count) in enumerate(result.delete, 1):
        dtl_file.write(f"{index:4d}:{count:5d}  ->  {char}\n")

    met = AsrMetricPerUtt(std_trn_path, reg_trn_path)
    met.gen_most_wrong(self.__most_wrong, num_most_wrong)

  @abstractmethod
  def from_utts(self, utts, decode_trn_path, num_most_wrong=20,
                word_phases=None):
    """生成标准的评测结果文件夹, 标准结果从utterance列表获取,
       识别结果从传入的decode trn文件获取.

    Args:
      utts: 标准结果的utterance列表.
      decode_trn_path: 解码的trn文件路径.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[string1, string2,,,],,,], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """

  @abstractmethod
  def from_trn_map(self, std_trn_map, reg_trn_map, num_most_wrong=20,
                   word_phases=None):
    """生成标准的评测结果文件夹, 标准结果和识别结果均从trn id到以字为单位的文本的映射获取.

    Args:
      std_trn_map: 标准结果的trn id到以词为单位的文本的映射.
      reg_trn_map: 识别结果的trn id到以词为单位的文本的映射.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[string1, string2,,,],,,], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """

  def std_trn_map(self):
    """获取标准结果trn id到以词为单位的文本的映射.

    Returns:
      标准结果trn id到以词为单位的文本的映射.
    """
    return get_trn_map(self._std_trn_path, SplitType.WORD)

  def reg_trn_map(self):
    """获取识别结果trn id到以词为单位的文本的映射.

    Returns:
      标准识别trn id到以词为单位的文本的映射.
    """
    return get_trn_map(self._reg_trn_path, SplitType.WORD)

  def most_wrong(self):
    """获取most wrong文件内的内容.

    Returns:
      most wrong文件内的内容列表, 元素(trn id, content lines).
    """
    lines = get_lines_from_file(self.__most_wrong)

    results = list()
    num = 8  # most wrong文件内每个trn id内容的行数.
    for i in range(0, len(lines), num):
      trn_id = lines[i].rsplit("trn_id: ", 1)[1]
      results.append((trn_id, lines[i: i + num]))
    return results


class CnAsrEvalDir(AsrEvalDirBase):
  """中文Asr评测结果文件夹.

  Attributes:
    __std_char_trn_path: 标准结果以字为单位的trn id到文本的映射.
    __reg_char_trn_path: 识别结果以字为单位的trn id到文本的映射.
    _dtl_path: 评测结果文件路径, 包含错误率等信息.
    __english_words: 从发音词典获取的英文词集合.
  """

  def __init__(self, eval_dir, language="zh"):
    """初始化.

    Args:
      eval_dir: Asr评测结果文件夹类.
      language: 语种, 可选值: zh,代表中文;en,代表英文;yue,代表粤语;zh-en,代表中英混合. 默认zh.
    """
    super(CnAsrEvalDir, self).__init__(eval_dir)
    self.__std_char_trn_path = eval_dir / "std_char.trn"
    self.__reg_char_trn_path = eval_dir / "reg_char.trn"
    self._dtl_path = eval_dir / "char.dtl"
    if language == "yue":
      dict_path = CANTONESE_DICT_PATH
    else:
      # pylint: disable=fixme
      dict_path = DICT_PATH # TODO(cui): 修改
    self.__english_words = PronunciationDict(dict_path).english_words

  def __get_english_chars(self, en_str):
    """得到英文的字列表.

    如果英文字符串在发音词典的英文单词集合内, 返回只包含该英文字符串的列表,
    否则返回字母列表, 输入的英文字符串为空返回空列表.

    Args:
      en_str: 英文字符串, 可以为空.

    Returns:
      英文的字列表.
    """
    if en_str:
      return [en_str] if en_str in self.__english_words else list(en_str)
    else:
      return list()

  def __get_chars_from_words(self, words):
    """从词语列表中获取字列表, 词语列表可能包含中英文混合的情况.

    Args:
      words: 词语列表.

    Returns:
      字列表.
    """
    chars = list()
    for word in words:
      if word.startswith(("{", "<")):
        chars.append(word)
      else:
        en_str = ""
        for char in word:
          if char.encode("utf-8").isalpha():
            en_str += char
          else:
            chars += self.__get_english_chars(en_str) + [char]
            en_str = ""

        chars += self.__get_english_chars(en_str)  # 结尾为英文的情况.
    return chars

  def __convert_trn_to_char(self, trn_path, out_char_trn_path):
    """将以词为单位的trn文件转成以字为单位的trn文件.

    Args:
      trn_path: 以词为单位的trn文件路径.
      out_char_trn_path: 生成的以字为单位的trn文件路径.
    """
    with out_char_trn_path.open("w", encoding="utf-8") as char_trn_file:
      for line in get_lines_from_file(trn_path):
        *words, trn_id = line.split()
        chars = self.__get_chars_from_words(words)
        char_trn_file.write(f"{' '.join(chars)} {trn_id}\n")

  def from_utts(self, utts, decode_trn_path, num_most_wrong=20,
                word_phases=None):
    """生成标准的评测结果文件夹, 标准结果从utterance列表获取, 识别结果从传入的decode trn文件获取.

    Args:
      utts: 标准结果的utterance列表.
      decode_trn_path: 解码的trn文件路径.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[string1, string2,,,],,,], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    if not decode_trn_path.exists():  # 完全没有识别出文本的情况, 可能为空的trn文件.
      reg_trn_map = dict()
    else:
      reg_trn_map = get_trn_map(decode_trn_path, SplitType.WORD)

    write_trn_trx_list(list(reg_trn_map.items()), self._reg_trn_path)
    self.__convert_trn_to_char(self._reg_trn_path, self.__reg_char_trn_path)

    get_std_trn_from_utts(utts, self.__std_char_trn_path)
    get_std_trn_from_utts(utts, self._std_trn_path, False)
    self._gen_dtl_most_wrong(self.__std_char_trn_path, self.__reg_char_trn_path,
                             SplitType.CHAR, num_most_wrong,
                             word_phases=word_phases)

  def from_trn_map(self, std_trn_map, reg_trn_map, num_most_wrong=20,
                   word_phases=None):
    """生成标准的评测结果文件夹, 标准结果和识别结果均从trn id到以字为单位的文本的映射获取.

    Args:
      std_trn_map: 标准结果的trn id到以词为单位的文本的映射.
      reg_trn_map: 识别结果的trn id到以词为单位的文本的映射.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[词1, 词2,...],...], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    write_trn_trx_list(std_trn_map.items(), self._std_trn_path)
    write_trn_trx_list(reg_trn_map.items(), self._reg_trn_path)
    self.__convert_trn_to_char(self._std_trn_path, self.__std_char_trn_path)
    self.__convert_trn_to_char(self._reg_trn_path, self.__reg_char_trn_path)
    self._gen_dtl_most_wrong(self.__std_char_trn_path, self.__reg_char_trn_path,
                             SplitType.CHAR, num_most_wrong,
                             word_phases=word_phases)

  def from_char_trn_map(self, std_char_trn_map, reg_char_trn_map,
                        num_most_wrong=20, word_phases=None):
    """生成标准的评测结果文件夹, 标准结果和识别结果均从trn id到以字为单位的文本的映射获取.

    Args:
      std_char_trn_map: 标准结果的trn id到以字为单位的文本的映射.
      reg_char_trn_map: 识别结果的trn id到以字为单位的文本的映射.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[词1, 词2,...],...], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    write_trn_trx_list(std_char_trn_map.items(), self.__std_char_trn_path)
    write_trn_trx_list(reg_char_trn_map.items(), self.__reg_char_trn_path)
    self._gen_dtl_most_wrong(self.__std_char_trn_path, self.__reg_char_trn_path,
                             SplitType.CHAR, num_most_wrong,
                             word_phases=word_phases)

  def std_char_trn_map(self):
    """获取标准结果trn id到以字为单位的文本的映射.

    Returns:
      标准结果trn id到以字为单位的文本的映射.
    """
    return get_trn_map(self.__std_char_trn_path, SplitType.WORD)

  def reg_char_trn_map(self):
    """获取识别结果trn id到以字为单位的文本的映射.

    Returns:
      标准识别trn id到以字为单位的文本的映射.
    """
    return get_trn_map(self.__reg_char_trn_path, SplitType.WORD)


class EnAsrEvalDir(AsrEvalDirBase):
  """英文Asr评测结果文件夹.

  Attributes:
    __reg_norm_trn_path: 识别结果对齐标签结果中be动词缩写后的文本.
    _dtl_path: 评测结果文件路径, 包含错误率等信息.
  """

  def __init__(self, eval_dir):
    """初始化.

    Args:
      eval_dir: Asr评测结果文件夹类.
    """
    super(EnAsrEvalDir, self).__init__(eval_dir)
    self.__reg_norm_trn_path = eval_dir / "reg_norm.trn"
    self._dtl_path = eval_dir / "word.dtl"

  def from_utts(self, utts, decode_trn_path, num_most_wrong=20,
                word_phases=None):
    """生成标准的评测结果文件夹, 标准结果从utterance列表获取,
       识别结果从传入的decode trn文件获取.

    Args:
      utts: 标准结果的utterance列表.
      decode_trn_path: 解码的trn文件路径.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[string1, string2,,,],,,], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    if not decode_trn_path.exists():  # 完全没有识别出文本的情况, 可能为空的trn文件.
      reg_trn_map = dict()
    else:
      reg_trn_map = get_trn_map(decode_trn_path, SplitType.WORD)

    # 对齐英文中的be动词缩写
    norm_reg_trn_map = norm_trn_english(reg_trn_map)
    write_trn_trx_list(list(norm_reg_trn_map.items()), self.__reg_norm_trn_path)

    _get_std_trn_from_utts_and_norm(utts, self._std_trn_path, False)
    self._gen_dtl_most_wrong(self._std_trn_path, self.__reg_norm_trn_path,
                             SplitType.WORD, num_most_wrong,
                             word_phases=word_phases)

  def from_trn_map(self, std_trn_map, reg_trn_map, num_most_wrong=20,
                   word_phases=None):
    """生成标准的评测结果文件夹, 标准结果和识别结果均从trn id到以词为单位的文本的映射获取.

    Args:
      std_trn_map: 标准结果的trn id到以词为单位的文本的映射.
      reg_trn_map: 识别结果的trn id到以词为单位的文本的映射.
      num_most_wrong: 生成的错误字数错误最多的句子, 默认20.
      word_phases: [[string1, string2,,,],,,], 列表中的每个列表表示一个词组信息,
                   用每个词组将识别结果和标注结果进行对比, 统计词组准确率和错误率.
    """
    norm_std_trn_map = norm_trn_english(std_trn_map)
    write_trn_trx_list(norm_std_trn_map.items(), self._std_trn_path)
    norm_reg_trn_map = norm_trn_english(reg_trn_map)
    write_trn_trx_list(list(norm_reg_trn_map.items()), self.__reg_norm_trn_path)
    self._gen_dtl_most_wrong(self._std_trn_path, self.__reg_norm_trn_path,
                             SplitType.WORD, num_most_wrong,
                             word_phases=word_phases)


def _get_std_trn_from_utts_and_norm(utterances, trn_path, char_based=True):
  """从utterance的列表生成正规化后的标准trn文件.
  由于英文数据库中部分数据还没有正规化, 因此对标准数据也进行正规化的流程.

  Args:
    utterances: utterance的列表.
    trn_path: 生成的trn文件路径.
    char_based: 是否以字为单位， 默认为True.
  """
  trn_id_trx_list = list()
  for utt in utterances:
    trx = " ".join(_get_chars_from_trx(utt.text)) if char_based else utt.text
    trn_id_trx_list.append((utt.utt_id(), trx))
  trn_id_trx_map = dict()
  for line in trn_id_trx_list:
    trn_id_trx_map[line[0]] = line[1]
  norm_trn_id_trx_map = norm_trn_english(trn_id_trx_map)
  write_trn_trx_list(norm_trn_id_trx_map.items(), trn_path)


def get_std_trn_from_utts(utterances, trn_path, char_based=True):
  """从utterance的列表生成标准的trn文件.

  Args:
    utterances: utterance的列表.
    trn_path: 生成的trn文件路径.
    char_based: 是否以字为单位， 默认为True.
  """
  trn_id_trx_list = list()
  for utt in utterances:
    trx = " ".join(_get_chars_from_trx(utt.text)) if char_based else utt.text
    trn_id_trx_list.append((utt.utt_id(), trx))
  write_trn_trx_list(trn_id_trx_list, trn_path)


def get_trn_map(trn_path, split_type):
  """得到trn_id到句子的Dict.

  Args:
    trn_path: trn文件的路径.
    split_type: 控制如何对句子切分的枚举类型.

  Returns:
    trn_id -> 句子 的Dict.
  """
  trn_dict = dict()

  for line in get_lines_from_file(trn_path):
    sentence, trn_id = line[:-1].split("(")
    sentence = sentence.strip()  # 处理trn文件中括号和文本之间可能存在的空格.

    if split_type == SplitType.NO:
      trn_dict[trn_id] = "".join(sentence.split())
    elif split_type == SplitType.CHAR:
      trn_dict[trn_id] = " ".join(_get_chars_from_trx(sentence))
    elif split_type == SplitType.WORD:
      trn_dict[trn_id] = sentence
    else:
      raise ValueError(f"不支持的切分类型{split_type}.")

  return trn_dict


def norm_trn_english(trn_map):  # pylint: disable=too-many-branches
  """规范识别文本或标准结果, 包括去标点, 删除场景标注, 对be动词缩写归一化.

  Args:
    trn_map: 词为单位的trn_id对文本的映射, 来自识别文本或者标准结果.

  Returns:
    规范be动词缩写后的结果, trn_id对文本的映射.
  """
  norm_trn_map = dict()
  letter_dot_pattern = re.compile(r"([a-z]+\.)")
  for trn_id, text in trn_map.items():
    result = re.findall(r"(\[.*?\])", text)  # 删除噪声识别标记
    for res in result:
      text = text.replace(res, "")

    text = text.replace('[vocalized-noise]', "")
    text = text.replace('<unk>', "")
    text = text.replace('%hesitation', "")

    text = text.lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("?", "")
    text = text.replace("!", "")
    text = text.replace(".", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace('"', "")
    text = ' '.join(text.split())
    text = ' '.join(text.split('-'))

    # 此处对be动词的缩写进行归一化，目的是减少不同数据集中不同标注的影响.
    # 这种替换不保证语义正确，仅在评测中使用.
    # 例如，No I can't 不能写成 No I can not，但在评测时暂且认为等价.
    # 在统一数据集的标注后，可以调整此逻辑.
    text = text.replace(" '", "'")
    if "let's" in text:
      text = text.replace("let's", "let us")
    if "i'm" in text:
      text = text.replace("i'm", "i am")
    if "'s been" in text:
      text = text.replace("'s been", " has been")
    if "'s" in text:
      text = text.replace("'s", " is")
    if "'re" in text:
      text = text.replace("'re", " are")
    if "can't" in text:
      text = text.replace("can't", "can not")
    if "won't" in text:
      text = text.replace("won't", "will not")
    if "n't" in text:
      text = text.replace("n't", " not")
    if "'d" in text:
      text = text.replace("'d", " would")
    if "'ve" in text:
      text = text.replace("'ve", " have")
    if "'ll" in text:
      text = text.replace("'ll", " will")

    for letter_dot in re.findall(letter_dot_pattern, text):
      text = text.replace(letter_dot, letter_dot[0:-1])  # p. -> p

    norm_trn_map[trn_id] = text

  return norm_trn_map


def get_sentence_split(sentence, sent_formator):
  """将一个完整句子分词后, 然后按照不同的粒度组装.

  Args:
    sentence: 一句连续的句子.
    sent_formator: SentFormatter对象.

  Returns:
    重新组装好的句子, 每个粒度词之间用空格连接.
  """
  # 词(英文词, 中文词)级别.
  sentence = sent_formator.format_without_filter_oovs(sentence)
  return " ".join(_get_chars_from_trx(sentence))


def _get_chars_from_trx(transcript):
  """从文本中获取转译的字符列表.

  Args:
    transcript: 转译结果.

  Returns:
    转译结果中的字符列表.
  """
  characters = list()

  for words in transcript.split():
    if words.startswith("{") or words.encode("utf-8").isalpha():
      characters.append(words)
    else:
      characters += list(words)
  return characters


def __trn_key(trn_id_and_transcript):
  """生成可比较的trn键值.

  Args:
    trn_id_and_transcript: trn_id和transcript组成的元组.

  Returns:
    可比较的trn键值.
  """

  filename, begin_time, *_ = trn_id_and_transcript[0].split("-")[-4:]
  return filename, int(begin_time)


def write_trn_trx_list(trn_transcript_list, output_path):
  """把trn id和转译结果组成的列表写入到output_path, 结果保持相同文件的内容在一起, 并按照时间排序.

  Args:
    trn_transcript_list: 形如[(trn_id, transcript)...]
    output_path: 输出的文件名.
  """
  sorted_trn_transcript_list = sorted(trn_transcript_list, key=__trn_key)

  with output_path.open("w", encoding="utf-8") as trn_file:
    for utt_id, transcript in sorted_trn_transcript_list:
      trn_file.write(f"{transcript} ({utt_id})\n")


def get_cer(eval_result_dir, language="zh"):
  """从评测结果文件夹内获取cer. 默认获取中文cer, 支持传参获取英文wer.

  Args:
    eval_result_dir: 评测结果文件夹.
    language: 语种, 可选值: zh,代表中文;en,代表英文;yue,代表粤语;zh-en,代表中英混合. 默认zh.

  Returns:
    中文返回cer, 英文返回wer.
  """
  if language == "en":
    wer_line = get_lines_from_file(eval_result_dir / "word.dtl")[2]
  else:
    wer_line = get_lines_from_file(eval_result_dir / "char.dtl")[2]
  return float(__WER_PATTERN.findall(wer_line)[0])
