
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .utils import to_categorical

dtype = torch.FloatTensor

from . import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, vocab_size, sparsity, freeze, criterion):
        super(ChildSumTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.node_forward = ChildSumNodeLSTM(in_dim, mem_dim, vocab_size)
        self.criterion = criterion
        self.mem_dim = mem_dim
        self.vocab_size = vocab_size

    def forward(self, tree, loss):
        inputs = tree.tensor
        for idx in range(len(tree.children)):
            self.forward(tree.children[idx], loss)
        inputs_emd = self.emb(inputs)
        if len(tree.children) == 0:
            child_c = inputs_emd[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs_emd[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        c, h, output = self.node_forward(inputs_emd, child_c, child_h)
        inputs_test = torch.tensor(to_categorical(inputs, self.vocab_size), dtype=torch.float64)
        output = output.to(torch.float64)
        # print(inputs.dtype, output.dtype)
        node_loss = self.criterion(inputs_test.view(1, -1), output.view(1, -1))
        loss += node_loss
        tree.state = c, h
        return tree.state


class ChildSumNodeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, vocab_size):
        super(ChildSumNodeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.out = nn.Linear(self.mem_dim, vocab_size)

    def forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        out = self.out(h)
        soft = torch.softmax(out, dim=1)
        # values, output = torch.max(soft, 1)
        # print(values, output)
        return c, h, soft


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, vocab_size, sparsity, freeze, criterion):
        super(Classifier, self).__init__()
        self.child_lstm = ChildSumTreeLSTM(in_dim, mem_dim, vocab_size, sparsity, freeze, criterion)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(mem_dim, mem_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(mem_dim, out_dim),
            nn.Softmax(dim=0),
        )

    def forward(self, tree, loss):
        c, h = self.child_lstm(tree, loss)
        out = self.classifier(h)
        # output, index = torch.max(out, 1)
        return out


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, num_vocab, num_classes, n_hidden):
        super(BiLSTM_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_vocab = num_vocab
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.embedding = nn.Embedding(num_vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        input = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(
            torch.zeros(1 * 2, len(X), self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(
            torch.zeros(1 * 2, len(X), self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # print(pe.shape)
        # print(pe[:, 0::2].shape)
        # print(pe[:, 1::2].shape)
        # print(torch.sin(position * div_term).shape)
        # print(torch.cos(position * div_term).shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_vocab, num_classes, n_hidden, nhead, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(num_vocab, embedding_dim)
        self.ninp = embedding_dim
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]

    def classify(self, input):
        # input = self.embedding(X)  # input : [batch_size, len_seq, embedding_dim]
        input = input.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(
            torch.zeros(1 * 2, input.shape[1], self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(
            torch.zeros(1 * 2, input.shape[1], self.n_hidden))  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention  # model : [batch_size, num_classes], attention : [batch_size, n_step]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * torch.sqrt(self.ninp)
        # print(src.shape)
        src = self.pos_encoder(src)
        # print(src.shape)
        output = self.transformer_encoder(src, self.src_mask)
        return self.classify(output)
