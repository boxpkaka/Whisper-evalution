from typing import List, Dict
import json

def get_file(path: str) -> List:
    with open(path, 'r', encoding='utf-8') as f:
        file = f.readlines()
    f.close()
    file = [line.strip() for line in file]
    return file


def save_file(path: str, file: List) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for item in file:
            f.write(item + '\n')
    f.close()


def get_json(path: str) -> Dict:
    with open(path, 'r') as f:
        file = json.load(f)
    return file