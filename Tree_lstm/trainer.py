from tqdm import tqdm

import torch
from torch.autograd import Variable
from .utils import convert_tree
from sklearn.metrics import f1_score
import numpy as np

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, batchsize, type):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        self.batchsize = batchsize
        self.type = type

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        if self.type == 'tree':
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                tree, label = dataset[idx]
                tree = convert_tree(tree, self.device)
                label = torch.tensor(self.to_categorical(label, 2), dtype=torch.float64).to(self.device)
                input = torch.randn(3, 5, requires_grad=True)
                target = torch.randn(3, 5)
                loss = self.criterion(input, target)
                output = self.model(tree, loss)
                # print(output.shape, label.shape)
                class_loss = self.criterion(output.to(torch.float64).view(1, -1), label.to(torch.float64).view(1, -1))
                loss += class_loss
                total_loss += loss.item()
                loss.backward()
                if idx % self.batchsize == 0 and idx > 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            self.epoch += 1
            return total_loss / len(dataset)
        else:
            for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
                bug, method, label = dataset[idx]
                # print(label)
                train_bug = Variable(bug.to(self.device))
                train_method = Variable(method.to(self.device))
                train_label = Variable(torch.tensor(self.to_categorical(label, 2), dtype=torch.float64).view(1, -1)
                                       .to(self.device))
                output, _ = self.model(train_bug, train_method)
                # output = self.model(train_batch)
                # print(output[0], label)
                # print(output.shape)

                loss = self.criterion(output.to(torch.float64), train_label)
                total_loss += loss.item()
                loss.backward()
                if idx % self.batchsize == 0 and idx > 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # print('Epoch:', '%04d' % (self.epoch + 1), 'cost =', '{:.6f}'.format(loss))
            self.epoch += 1
            return total_loss / len(dataset)


    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        if self.type == 'tree':
            with torch.no_grad():
                total_loss = 0.0
                predictions = []
                indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
                for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                    tree, label = dataset[idx]
                    tree = convert_tree(tree, self.device)
                    label = torch.tensor(self.to_categorical(label, 2), dtype=torch.float64).to(self.device)
                    input = torch.randn(3, 5, requires_grad=True)
                    target = torch.randn(3, 5)
                    loss = self.criterion(input, target)
                    output = self.model(tree, loss)
                    class_loss = self.criterion(output.to(torch.float64).view(1,-1), label.to(torch.float64).view(1,-1))
                    loss += class_loss
                    total_loss += loss.item()
                    output = output.squeeze().to('cpu')
                    predictions.append(output.numpy())
            return total_loss / len(dataset), predictions
        else:
            with torch.no_grad():
                total_loss = 0.0
                predictions = []
                indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
                for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                    bug, method, label = dataset[idx]
                    # print(label.shape)
                    train_bug = Variable(bug.to(self.device))
                    train_method = Variable(method.to(self.device))
                    train_label = Variable(torch.tensor(self.to_categorical(label, 2), dtype=torch.float64).view(1, -1)
                                           .to(self.device))
                    output, _ = self.model(train_bug, train_method)
                    # output = self.model(train_batch)
                    # print(output[0], label)
                    # print(output.shape)
                    loss = self.criterion(output.to(torch.float64), train_label)
                    total_loss += loss.item()
                    _, output = torch.max(output.squeeze(), dim=0)
                    predictions.append(int(output.to('cpu').numpy()))
            return total_loss / len(dataset), predictions
