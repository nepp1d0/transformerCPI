# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:36
@author: LiFan Chen
@Filename: model.py
@Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead

import torch.nn as nn

from transformers import TransformerBlock
from embedding.bert import BERTEmbedding
from classification_head import ClassificationHead

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    src: https://github.com/codertimo/BERT-pytorch.git
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.cls_head = ClassificationHead(batch_size=8, hidden_size=self.hidden)


    def forward(self, x):
        # x = torch.Tensor(
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        #print("Length of input packed: ",torch.FloatTensor(x).shape)
        #mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        #print(torch.FloatTensor(x[0]).shape)

        # Comcatenate smiles and target sequence
        input_seq = torch.cat((x[0], x[2]), dim=1)
        #print(f'input_seq shape is: {input_seq.size()}')
        #torch.cat(x, out=)
        # Concatenate masks of smiles and target
        input_masks = torch.cat((x[1], x[3]), dim=1)

        # Casting to match nn.Embedding requirements
        x_in = torch.tensor(input_seq).to(torch.int64)
        x = self.embedding(x_in)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            #print(f'Feeding transformers layers with input of size {x.size()} and masks of {input_masks.size()}')
            #  x = torch.Tensor([input_seq_len, hidden_dim, ])
            x = transformer.forward(x, input_masks)
        #print(f"Output of bert for loss is : {x.shape}")
        x = self.cls_head(x)
        return x

def pack(smiles, targets, labels, device):
    smiles_len = 0
    targets_len = 0
    N = len(smiles)

    smile_num = []
    for smile in smiles:
        smile_num.append(smile.shape[0])
        if smile.shape[0] >= smiles_len:
            smiles_len = smile.shape[0]

    target_num = []
    for target in targets:
        target_num.append(target.shape[0])
        if target.shape[0] >= targets_len:
            targets_len = target.shape[0]

    smiles_new = torch.zeros((N, smiles_len, 100), device=device)
    i = 0
    for smile in smiles:
        a_len = smile.shape[0]
        smiles_new[i, :a_len, :] = smile
        i += 1

    targets_new = torch.zeros((N, targets_len, 100), device=device)
    i = 0
    for target in targets:
        a_len = target.shape[0]
        targets_new[i, :a_len, :] = target
        i += 1

    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    return (smiles_new, targets_new, labels_new, smile_num, target_num)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        smiles, smiles_masks, targets, targets_masks, labels = torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device)
        for data in dataset:
            i = i+1
            smile, smile_mask, target, target_mask, label = data
            if smiles.size() == torch.Size([0]):
                smiles = torch.cat((smiles, smile))
            else:
                smiles = torch.vstack((smiles, smile))

            if smiles_masks.size() == torch.Size([0]):
                smiles_masks = torch.cat((smiles_masks, smile_mask))
            else:
                smiles_masks = torch.vstack((smiles_masks, smile_mask))

            if targets.size() == torch.Size([0]):
                targets = torch.cat((targets, target))
            else:
                targets = torch.vstack((targets, target))

            if targets_masks.size() == torch.Size([0]):
                targets_masks = torch.cat((targets_masks, target_mask))
            else:
                targets_masks = torch.vstack((targets_masks, target_mask))

            if i % 8 == 0 or i == N:
                #data_pack = pack(smiles, smiles_mask, targets, targets_mask, labels, device)
                #print(f'Proteins in data pack have shape: {data_pack[1].shape}')
                pred = self.model((smiles, smiles_masks, targets, targets_masks, labels))
                loss = torch.nn.functional.mse_loss(pred, label)
                # loss = loss / self.batch
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                smiles, smiles_masks, targets, targets_masks, labels = torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device), torch.LongTensor(device=device)
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms,adjs,proteins, labels, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, PRC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
