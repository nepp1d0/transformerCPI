# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/9/17 8:54
@author: LiFan Chen
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import random
import os
import time
import timeit
import warnings

from model_bert import BERT
from model_bert import Trainer

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "GPCR_train"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('tokenizer_directory/' )
    smiles = load_tensor(dir_input + 'smiles', torch.FloatTensor)
    targets = load_tensor(dir_input + 'targets', torch.FloatTensor)
    labels = load_tensor(dir_input + 'labels', torch.FloatTensor)

    print(f'Shape of smiles loaded tensors {smiles[0].shape}')
    print(f'Shape of smiles masks loaded tensors {smiles[1].shape}')

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(smiles[0], smiles[1], targets[0], targets[1], labels))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_dev = split_dataset(dataset, 0.8)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 100
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 300
    kernel_size = 7

    print("Building BERT model")
    model = BERT(70 - 5, hidden=hid_dim, n_layers=n_layers, attn_heads=n_heads)  # 70 (smiles_vocab_size + targets_vocab_size) - 5 (special_tokens not repeated)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    #tester = Tester(model)

    """Output files."""
    file_AUCs = 'output_bert/result/bertv1' + '.txt'
    file_model = 'output_bert/model/' + 'bertv1'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev')
    if not os.path.exists('output_bert'):
        os.makedirs('output_bert')
    if not os.path.exists('output_bert/result'):
        os.makedirs('output_bert/result')
    if not os.path.exists('output_bert/model'):
        os.makedirs('output_bert/model')

    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()
    max_AUC_dev = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device)
        #AUC_dev, PRC_dev = tester.test(dataset_dev)

        end = timeit.default_timer()
        time = end - start

        #tester.save_AUCs(AUCs, file_AUCs)
        AUCs = [epoch, time, loss_train]
        print('\t'.join(map(str, AUCs)))
        '''if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
        print('\t'.join(map(str, AUCs)))'''

