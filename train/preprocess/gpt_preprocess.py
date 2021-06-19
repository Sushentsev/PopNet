import os
import argparse

DATA_PATH = 'data/gensongs/'
SAVE_PATH = 'train/preprocess/gpt_data/'


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path to folder with songs")
    parser.add_argument("save_path", type=str, help="where to save data files")
    args = parser.parse_args()

    preprocesser = GPTPreprocess(args.data_path, args.save_path)
    preprocesser.preprocess()
