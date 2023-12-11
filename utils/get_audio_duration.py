import soundfile

def get_duration_from_idx(idx: str) -> float:
    splited = idx.split('-')
    ed = float(splited[-2])
    bg = float(splited[-3])

    return round((ed - bg) * 0.001, 3)


if __name__ == '__main__':
    idx = '1000Cantonese00000978-0000000-0004715-S'

    print(get_duration_from_idx(idx))



