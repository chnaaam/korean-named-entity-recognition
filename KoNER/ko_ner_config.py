import os
import yaml


class ConfigBase:
    def __init__(self, data):
        for key, value in data.items():
            if type(value) is dict:
                self.__setattr__(key, ConfigBase(value))
            else:
                self.__setattr__(key, value)


def get_ko_ner_configuration(config_path, config_file):
    config_full_path = os.path.join(config_path, config_file)

    with open(config_full_path) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)

    return ConfigBase(data=data)
    # return ConfigBase(data={
    #     "dataset": ConfigBase(data=data["dataset"]),
    #     "tokenizer": ConfigBase(data=data["tokenizer"]),
    #     "model": ConfigBase(data=data["model"]),
    #     "parameters": ConfigBase(data=data["parameters"])
    # })
    #
