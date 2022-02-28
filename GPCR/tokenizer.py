from collections import defaultdict
import json
import os

class Tokenizer(object):
    '''
    Class which handle the creation of the vocabulary and the tokenization of dataset
    '''

    def __init__(self, max_len=None, tok_dir=None, load_from_file=False, purpose=None):
        """
            Init tokenizer, if tok_dir is specified, it loads pre-computed vocabulary .json
            otherwise just init the default_dict

                Parameters
                ----------
                max_len: int
                    maximum length of output sentences of tokenizer, if specified it automatically does padding and truncation
                    during pre_tokenization step
                tok_dir: string, path
                    path to .json file used to save/load

            """
        # Preprocess text
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.max_len = max_len
        self.TOKENIZER_DIRECTORY = tok_dir
        if purpose is not None:
            self.purpose = purpose
        def default_value():
            return '[UNK]'

        if load_from_file:
            try:
                with open(os.path.join(self.TOKENIZER_DIRECTORY, 'tokenizer.json'), 'r') as f:
                    self.word_dict = json.load(f)
                    self.vocab_size = len(self.word_dict)
            except IOError:
                # no such file, create an empty dictionary
                print('Please insert a valid path')
        else:
            self.word_dict = defaultdict(default_value)  # If word is not in vocabulary return [UNK]
            self.word_dict[self.unk_token] = 0
            self.word_dict[self.pad_token] = 1
            self.word_dict[self.cls_token] = 2
            self.word_dict[self.sep_token] = 3
            self.word_dict[self.mask_token] = 4

    def train(self,  text):
        """
            Train a blank tokenizer, which add word-ids to the word_dict default_dict,
            Call this method only if you don't have already initialize the tokenizer from folder
            since ids could be overwritten in different order

            Parameters
            ----------
            text: string
               Corpus to tokenize
            """
        # Compute word list
        text = ' '.join(text)
        word_list = list(set(" ".join(text).split()))
        # Compute word-id dictionary
        for i, w in enumerate(word_list):
            self.word_dict[w] = i + 5  # 5 = number of special tokens
        # number_dict = {i: w for i, w in enumerate(self.word_dict)}
        self.vocab_size = len(self.word_dict)
        with open(os.path.join(self.TOKENIZER_DIRECTORY, self.purpose + '_tokenizer.json'), 'w') as f:
            json.dump(self.word_dict, f)

    def tokenize(self, sentences):
        """
        Given list of token, returns list of ids of tokenized sentence

        Parameters
        ----------
        text: list of string
           Corpus to tokenize

        Returns
        ----------
        token_list: list of ids
        """
        token_list = list()
        masks_list = list()

        for sentence in sentences:
            arr = [self.word_dict[s] for s in sentence]
            token_list.append(arr)
            masks_list.append(self.get_attn_pad_mask(arr))
        return token_list, masks_list

    def pre_tokenize(self, sentences):
        """
        Given sentences, returns list of words with special tokens

        Parameters
        ----------
        text: list of string
           Corpus to tokenize

        Returns
        ----------
        token_list: list of tokens
        """
        token_list = list()
        for sentence in sentences:
            arr = [self.cls_token] + [s for s in sentence.split(' ')]
            if self.max_len:
                if len(arr) < self.max_len:
                    pads = [self.pad_token] * (self.max_len - len(arr) - 1)
                    arr = arr + pads
                else:
                    # Truncate
                    arr = arr[: self.max_len -1]
            arr = arr + [self.sep_token]
            token_list.append(arr)
        return token_list

    def get_attn_pad_mask(self, seq):
        '''
        source: https://colab.research.google.com/drive/13FjI_uXaw8JJGjzjVX3qKSLyW9p3b6OV?usp=sharing#scrollTo=s1PGksqBNuZM
        '''
        #mask = [0] * len(seq)
        # 1 is PAD token
        pad_attn_mask = [0 if e == 1 else 1 for e in seq]  # batch_size x 1 x len_k(=len_q), zero is masking padding tokens
        return pad_attn_mask