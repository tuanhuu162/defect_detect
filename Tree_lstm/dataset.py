import numpy as np
from tqdm import tqdm
from copy import deepcopy
from anytree.importer import JsonImporter, DictImporter
from anytree.iterators import PreOrderIter
from pandas import read_csv

import torch
import torch.utils.data as data

from Tree_lstm import Constants
from Tree_lstm.vocab import Vocab
from codecs import decode
import json




# Dataset class for elastic dataset
class ELASTICDataset(data.Dataset):
    def __init__(self, path, vocab, type="tree", mode="eager"):
        super(ELASTICDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = vocab.size()
        self.mode = 0 if mode == "eager" else 1
        self.type = 0 if type == "tree" else 1
        self.path = path
        if self.mode == 0:
            self.trees, self.labels = self.read_trees(path, self.type)
        else:
            self.trees = []
            self.labels = []
            data = read_csv(self.path, sep="|")
            for i in tqdm(range(len(data))):
                self.labels.append(data.loc[i, 'label'])
            # self.labels = torch.tensor(self.labels, dtype=torch.int64)
        # self.size = self.labels.size()
        self.size = len(self.labels)

    def __len__(self):
        # return self.size[0]
        return self.size

    def __getitem__(self, index):
        if self.mode == 0:
            tree = deepcopy(self.trees[index])
        else:
            data = read_csv(self.path, sep="|")
            tree = self.read_tree(data.loc[index, 'data'], self.type)
        label = deepcopy(self.labels[index])
        return tree, label

    def read_trees(self, file, type):
        data = read_csv(file, sep="|")
        trees = []
        labels = []
        for i in tqdm(range(len(data))):
            trees.append(self.read_tree(data.loc[i, 'data'], type))
            labels.append(data.loc[i, 'label'])
        # labels = torch.tensor(labels, dtype=torch.int64)
        return trees, labels

    def read_tree(self, data, type):
        importer = JsonImporter()
        root = importer.import_(data)
        # print(root)
        preorder_file = []
        for node in PreOrderIter(root):
            preorder_file.append(node.name)
            node.tensor = self.read_node(node)
            # print(node)
        indices = self.vocab.convertToIdx(preorder_file, Constants.UNK_WORD)
        vector = torch.tensor(indices, dtype=torch.long, device='cpu').view((1, -1))
        # return root
        return root if type == 0 else vector

    def read_node(self, node):
        indices = self.vocab.convertToIdx([node.name], Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu').view(1, -1)

class OwaspDataset(data.Dataset):
    def __init__(self, path, vocab, type="tree", mode="eager"):
        super(OwaspDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = vocab.size()
        self.mode = 0 if mode == "eager" else 1
        self.type = 0 if type == "tree" else 1
        self.path = path
        if self.mode == 0:
            self.bugs, self.methods, self.labels = self.read_trees(path, self.type)
        else:
            self.bugs = []
            self.methods = []
            self.labels = []
            with open(path) as f:
                data = json.loads(f.read())
            for i in tqdm(range(len(data['label']))):
                self.labels.append(data['label'][i])
            # self.labels = torch.tensor(self.labels, dtype=torch.int64)
        # self.size = self.labels.size()
        self.size = len(self.labels)

    def __len__(self):
        # return self.size[0]
        return self.size

    def __getitem__(self, index):
        if self.mode == 0:
            bug = deepcopy(self.bugs[index])
            method = deepcopy(self.methods[index])
        else:
            data = read_csv(self.path, sep="|")
            bug = self.read_tree(data['bug'][index], self.type)
            method = self.read_tree(data['method'][index], self.type)
        label = deepcopy(self.labels[index])
        return bug, method, label

    def read_trees(self, file, type):
        with open(file) as f:
            data = json.loads(f.read())
        bugs = []
        methods = []
        labels = []
        for i in tqdm(range(len(data['label']))):
            bugs.append(self.read_tree(data['bug'][i], type))
            methods.append(self.read_tree(data['method'][i], type))
            labels.append(data['label'][i])
        # labels = torch.tensor(labels, dtype=torch.int64)
        return bugs, methods, labels

    def read_tree(self, data, type):
        importer = JsonImporter()
        root = importer.import_(data)
        # print(root)
        preorder_file = []
        for node in PreOrderIter(root):
            preorder_file.append(node.name)
            node.tensor = self.read_node(node)
            # print(node)
        indices = self.vocab.convertToIdx(preorder_file, Constants.UNK_WORD)
        vector = torch.tensor(indices, dtype=torch.long, device='cpu').view((1, -1))
        # return root
        return root if type == 0 else vector

    def read_node(self, node):
        indices = self.vocab.convertToIdx([node.name], Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu').view(1, -1)



if __name__=="__main__":
    # write unique words from all token files
    # get vocab object from vocab file previously written
    vocab = Vocab(filename="D:\\study\\thesis\\data\\detect_defect\\vocab",
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    # test = ELASTICDataset("D:\\study\\thesis\\data\\detect_defect\\test.csv", vocab, mode="eager", type="lstm")
    # print(len(test))
    # print(test[0][0].shape)
    test = OwaspDataset("D:\\study\\thesis\\data\\code\\defect_detect\\test_owasp.json", vocab, mode="eager", type="lstm")
    print(len(test))
    print(test[0][0].shape)