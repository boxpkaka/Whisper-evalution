import os
from typing import List


def get_file(path: str) -> List:
    with open(path, 'r') as f:
        file = f.readlines()
    file = [line.strip() for line in file]
    return file


def save_file(path: str, file: List) -> None:
    with open(path, 'w') as f:
        for item in file:
            f.write(item + '\n')
