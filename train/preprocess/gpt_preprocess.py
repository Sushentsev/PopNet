import os
import argparse
from omegaconf import OmegaConf
from argparse import ArgumentParser

CONFIG_PATH = 'train/configs/gpt_config.yaml'


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str, default=CONFIG_PATH)
    return arg_parser


class GPTPreprocess:
    def __init__(self, path: str, save_path: str, test_size: float = 0.2):
        self.__path = path
        self.__test_size = test_size
        self.__save_path = save_path

    def preprocess(self) -> None:
        files = os.listdir(self.__path)
        train_size = 1 - self.__test_size
        train_filenames, valid_filenames = files[:int(len(files) * train_size)], files[int(len(files) * train_size):]

        train_filepath = os.path.join(self.__save_path, 'train.txt')
        valid_filepath = os.path.join(self.__save_path, 'valid.txt')

        for current_filepath, filenames in [(train_filepath, train_filenames),
                                            (valid_filepath, valid_filenames)]:
            with open(current_filepath, 'w', encoding='utf8', errors='ignore') as current_file:
                # 1 : -4 because 1 to remove space and -4 to remove .txt
                names = [x.split('-')[1][1:-4] for x in filenames]
                file_paths = [os.path.join(self.__path, x) for x in filenames]

                for file, name in zip(file_paths, names):
                    with open(file, 'r', encoding='utf8', errors='ignore') as f:
                        current_file.write('<startsong>\n')
                        current_file.write('<songname> ' + name + '\n')
                        current_file.write('<songlyrics>\n')

                        for line in f:
                            current_file.write(line)

                        current_file.write('<endsong>\n')


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    config = OmegaConf.load(__args.config)

    preprocesser = GPTPreprocess(config.preprocess.raw_data, config.preprocess.train_data)
    preprocesser.preprocess()
