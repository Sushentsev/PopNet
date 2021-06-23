import os
import re
import pandas as pd
import numpy as np
from os.path import join, dirname
from tqdm import tqdm


def split(dataset_path: str, val_percent: float, test_percent: float):
    np.random.seed(42)
    df = pd.read_csv(dataset_path, sep="\t")

    perm = np.random.permutation(df.index)
    m = len(df.index)
    val_end = int(val_percent * m)
    test_end = int(test_percent * m) + val_end

    val = df.iloc[perm[:val_end]]
    test = df.iloc[perm[val_end:test_end]]
    train = df.iloc[perm[test_end:]]

    save_dir = dirname(dataset_path)
    train.to_csv(join(save_dir, "train.tsv"), sep="\t", index=False)
    val.to_csv(join(save_dir, "val.tsv"), sep="\t", index=False)
    test.to_csv(join(save_dir, "test.tsv"), sep="\t", index=False)


def make_dataset(files_dir: str, save_dir: str):
    file_names = os.listdir(files_dir)
    df = pd.DataFrame(columns=["Title", "Lyrics"])

    for file_name in tqdm(file_names):
        splitted = file_name.split(" - ")
        song_author, song_title = splitted[0], splitted[1:]
        song_title = " ".join(song_title)

        if "(" in song_title:
            song_title = song_title[:song_title.index("(")]  # Remove latin title
        else:
            song_title = song_title[:-4]  # Remove .txt

        if re.search("[a-zA-Z]", song_title) is not None:
            continue

        with open(join(files_dir, file_name)) as file:
            df = df.append({"Title": song_title, "Lyrics": file.read()}, ignore_index=True)

    df.to_csv(join(save_dir, "dataset.tsv"), sep="\t", index=False)
