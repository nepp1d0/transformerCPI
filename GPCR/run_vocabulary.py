from tokenizer import Tokenizer
import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    data = pd.read_csv('../data/GPCR_train.txt', sep=" ", header=None)
    data.columns = ['smiles', 'target', 'interact']

    targets = data.target.drop_duplicates() # 346 targets
    targets.to_csv(r'../data/GPCR_train_targets.txt', header=None, index=None, sep=' ')

    smiles = data.smiles.drop_duplicates()  # 5359 smiles
    smiles.to_csv(r'../data/GPCR_train_smiles.txt', header=None, index=None, sep=' ')

    with open("../data/GPCR_train_targets.txt","r") as f:
        targets_list = f.read().strip().split('\n')

    with open("../data/GPCR_train_smiles.txt", "r") as f:
        smiles_list = f.read().strip().split('\n')

    max_len = 10

    target_tokenizer = Tokenizer(targets_list, max_len)
    print(target_tokenizer.vocab_size)

    smiles_tokenizer = Tokenizer(smiles_list, max_len)
    print(smiles_tokenizer.vocab_size)

    to_tokenize_target = [' '.join(s) for s in targets_list]
    tokenized_target = target_tokenizer.pre_tokenize(to_tokenize_target[:4])
    print(tokenized_target)

    to_tokenize_smiles = [' '.join(s) for s in smiles_list]
    tokenized_smiles = smiles_tokenizer.pre_tokenize(to_tokenize_smiles[:4])
    print(tokenized_smiles)

    #print("VOCAB SIZE:", len(vocab))
    #vocab.save_vocab(args.output_path)