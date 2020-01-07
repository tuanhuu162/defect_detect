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
    data = read_csv('metadata.csv', index_col=[0])
    path = data['new_path']
    label = [1 if i > 0 else 0 for i in data['Number of bugs']]
    
    train_data, test_data, train_label, test_label = [], [], [], []
    indices = permutation(len(label))
    train_data = [path[i] for i in indices[:int(len(label)*0.8)]]
    test_data =


if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing elastic dataset')
    print('=' * 80)
    splitdata()

    # base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # data_dir = os.path.join(base_dir, 'data')
    # sick_dir = os.path.join(data_dir, 'elastic')
    # train_dir = os.path.join(sick_dir, 'train')
    # dev_dir = os.path.join(sick_dir, 'dev')
    # test_dir = os.path.join(sick_dir, 'test')
    # make_dirs([train_dir, dev_dir, test_dir])
    #
    # # get vocabulary
    # build_vocab(
    #     glob.glob(os.path.join(sick_dir, '*/*.toks')),
    #     os.path.join(sick_dir, 'vocab.txt'))
    # build_vocab(
    #     glob.glob(os.path.join(sick_dir, '*/*.toks')),
    #     os.path.join(sick_dir, 'vocab-cased.txt'),
    #     lowercase=False)