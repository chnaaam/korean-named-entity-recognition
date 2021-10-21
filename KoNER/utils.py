import os


def initialize_directories(config):
    make_directory(config.path.vocab)
    make_directory(config.path.label)
    make_directory(config.path.cache)
    make_directory(config.path.model)


def is_existed_directories(path):
    return os.path.isdir(path)

def make_directory(path):
    if not is_existed_directories(path):
        os.mkdir(path)