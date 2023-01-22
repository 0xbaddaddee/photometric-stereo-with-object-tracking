import os
from typing import List


def create_dir(dir_name: str) -> None:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def is_regular_file(file_name: str) -> bool:
    return os.path.isfile(file_name)


def exists(path: str) -> bool:
    return os.path.exists(path)


def get_dir_content(dir_name: str) -> List[str]:
    return sorted(os.listdir(dir_name))


def join_paths(*args):
    return os.path.join(*args)


def has_ext(file_name: str, file_extension):
    ext = file_name.rpartition('.')[-1]
    return ext == file_extension
