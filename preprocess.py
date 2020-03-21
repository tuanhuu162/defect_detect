"""
Preprocessing script for elastic data.
"""

import os
import glob
from utils.utils import preproces
from pandas import read_csv
from numpy.random import permutation

DATAPATH = "elastic/"

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def splitdata():
    train_data = read_csv('metadata.csv', index_col=[0])
    train_path = list(train_data['new_path'])
    train_label = [1 if i > 0 else 0 for i in train_data['Number of bugs']]
    train_hasbug, train_nothave = [], []
    print("split data.........................")
    for i in range(len(train_label)):
        if train_label[i] == 1:
            train_hasbug.append((train_path[i], train_label[i]))
        else:
            train_nothave.append((train_path[i], train_label[i]))

    print("split train test .......................")
    print(len(train_hasbug), len(train_nothave))
    indices_hasbug = permutation(1000)
    train_data = [train_hasbug[i] for i in indices_hasbug]
    # train_data = [hasbug[i] for i in indices_hasbug[:int(len(indices_hasbug)*0.8)]]
    # test_data = [hasbug[i] for i in indices_hasbug[int(len(indices_hasbug)*0.8):]]
    indices_nothas = permutation(1000 + 1000)

    train_data.extend([train_nothave[i] for i in indices_nothas])
    # train_data.extend([nothave[i] for i in indices_nothas[:int(len(indices_nothas)*0.8)]])
    # test_data.extend([nothave[i] for i in indices_nothas[int(len(indices_nothas)*0.8):]])
    # train_data.extend([nothave[i] for i in indices_hasbug[:int(len(indices_hasbug)*0.8)]])
    # test_data.extend([nothave[i] for i in indices_hasbug[int(len(indices_hasbug)*0.8):]])
    train_data = [train_data[i] for i in permutation(len(train_data))]
    test_data = read_csv('test_metadata.csv', index_col=[0])
    test_path = list(test_data['new_path'])
    test_label = [1 if i > 0 else 0 for i in test_data['bug']]
    test_hasbug, test_nothave = [], []
    print("split data.........................")
    for i in range(len(test_label)):
        if test_label[i] == 1:
            test_hasbug.append((test_path[i], test_label[i]))
        else:
            test_nothave.append((test_path[i], test_label[i]))

    print("split train test .......................")
    print(len(test_hasbug), len(test_nothave))
    test_data = [(test_path[i], test_label[i]) for i in permutation(len(test_data))]
    # if not os.path.exists("train"):
    #     os.mkdir("train")
    # if not os.path.exists("test"):
    #     os.mkdir("test")
    vocab = []
    print("make ast data.....................")
    with open('train.csv', 'w') as file:
        file.write("label|data\n")
        for i in train_data:
            exported = preproces(i[0], vocab, 'train')
            if exported == '':
                continue
            file.write(str(i[1]) + '|' + exported  + "\n")

    with open('test.csv', 'w') as file:
        file.write("label|data\n")
        for i in test_data:
            exported = preproces(i[0], vocab, 'train')
            if exported == '':
                continue
            file.write(str(i[1]) + '|' + exported + "\n")

    vocab = set(vocab)
    with open('vocab', 'w') as file:
        for i in vocab:
            file.write(i + "\n")

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing elastic dataset')
    print('=' * 80)
    splitdata()
    print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
