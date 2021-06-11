import os

path = 'PopNet/data/gensongs/'

files = os.listdir(path)
train_files, valid_files = files[:int(len(files) * 0.8)], files[int(len(files) * 0.8):]

names = [x.split('-')[1][:-4] for x in train_files]
files = [os.path.join(path, x) for x in train_files]

train_file = open('train.txt', 'w')

for file, name in zip(files, names):
    f = open(file, "r")
    train_file.write('<startsong>\n')
    train_file.write('<songname>' + name + '\n')
    train_file.write('<songlyrics>\n')

    for line in f:
        train_file.write(line)

    train_file.write('<endsong>\n')
    f.close()

train_file.close()

names = [x.split('-')[1][:-4] for x in valid_files]
files = [os.path.join(path, x) for x in valid_files]
valid_file = open('valid.txt', 'w')

for file, name in zip(files, names):
    f = open(file, "r")
    valid_file.write('<startsong>\n')
    valid_file.write('<songname>' + name + '\n')
    valid_file.write('<songlyrics>\n')

    for line in f:
        valid_file.write(line)

    valid_file.write('<endsong>\n')
    f.close()

valid_file.close()
