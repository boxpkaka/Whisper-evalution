import re

PATTERN_NUM = re.compile(r'[0-9]+\.[0-9]+|[0-9]+')
NUM_CHAR = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九", ".": "点"}


def num_to_char(line_text):
    """将文本中的数字转为汉字表示, 支持小数形式.

    Args:
    line_text: 文本字符串.

    Returns:
    数字转换为汉字后的文本(若有数字), 否则返回原文本.
    """

    if re.search(PATTERN_NUM, line_text):
        repl_text = []
        for num in re.findall(PATTERN_NUM, line_text):
            text = ''.join([NUM_CHAR[c] for c in num])
            repl_text.append((num, text))
        sorted_repl = sorted(repl_text, key=lambda x: len(x[0]), reverse=True)
        ans = line_text
        for orig, repl in sorted_repl:
            ans = ans.replace(orig, repl)
        return ans
    else:
        return line_text


if __name__ == '__main__':
    a = '2.3444'
    print(num_to_char(a))
