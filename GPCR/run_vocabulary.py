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

    targets = data.target # 346 targets
    targets.to_csv(r'../data/GPCR_train_targets.txt', header=None, index=None, sep=' ')

    smiles = data.smiles  # 5359 smiles
    smiles.to_csv(r'../data/GPCR_train_smiles.txt', header=None, index=None, sep=' ')

    labels = data.interact  # interactions
    labels.to_csv(r'../data/GPCR_train_labels.txt', header=None, index=None, sep=' ')

    with open("../data/GPCR_train_targets.txt", "r") as f:
        targets_list = f.read().strip().split('\n')

    with open("../data/GPCR_train_smiles.txt", "r") as f:
        smiles_list = f.read().strip().split('\n')

    with open("../data/GPCR_train_labels.txt", "r") as f:
        labels_list = f.read().strip().split('\n')
    labels_list = [[int(x)] for x in labels_list]
    print(labels_list)
    max_len = 100
    TOKENIZER_DIR = 'tokenizer_directory/'

    target_tokenizer = Tokenizer(max_len, TOKENIZER_DIR, load_from_file=False, purpose='target')
    target_tokenizer.train(targets_list)
    print(target_tokenizer.vocab_size)

    smiles_tokenizer = Tokenizer(34, TOKENIZER_DIR, load_from_file=False, purpose='smiles')
    smiles_tokenizer.train(smiles_list)
    print(smiles_tokenizer.vocab_size)

    to_tokenize_target = [' '.join(s) for s in targets_list]
    tokenized_target = target_tokenizer.pre_tokenize(to_tokenize_target)

    to_tokenize_smiles = [' '.join(s) for s in smiles_list]
    tokenized_smiles = smiles_tokenizer.pre_tokenize(to_tokenize_smiles)

    tokenized_target = target_tokenizer.tokenize(tokenized_target)
    tokenized_smiles = smiles_tokenizer.tokenize(tokenized_smiles)

    np.save(TOKENIZER_DIR + 'targets', tokenized_target, allow_pickle=True)
    np.save(TOKENIZER_DIR + 'smiles', tokenized_smiles, allow_pickle=True)
    np.save(TOKENIZER_DIR + 'labels', labels, allow_pickle=True)
    #print("VOCAB SIZE:", len(vocab))
    #vocab.save_vocab(args.output_path)