import re
from collections import defaultdict


class Tokenizer(object):
    '''
    Class which handle the creation of the vocabulary and the tokenization of dataset
    '''

    def __init__(self, text, max_len=None, min_freq=1):
        # Preprocess text
        #sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.max_len = max_len
        def default_value():
            return '[UNK]'
        self.word_dict = defaultdict(default_value)  # If word is not in vocabulary return [UNK]

        # Compute word list
        text = ' '.join(text)
        self.word_list = list(set(" ".join(text).split()))
        # Compute word-id dictionary
        self.word_dict[self.unk_token] = 0
        self.word_dict[self.pad_token] = 1
        self.word_dict[self.cls_token] = 2
        self.word_dict[ self.sep_token] = 3
        self.word_dict[self.mask_token] = 4
        for i, w in enumerate(self.word_list):
            self.word_dict[w] = i + 5  # 5 = number of special tokens

        #number_dict = {i: w for i, w in enumerate(self.word_dict)}
        self.vocab_size = len(self.word_dict)

    def tokenize(self, sentences):
        '''
        Given sentences, returns list of ids of tokenized sentence
        '''
        token_list = list()
        for sentence in sentences:
            arr = [self.word_dict[s] for s in ' '.join(sentence).split()]
            token_list.append(arr)
        return token_list

    def pre_tokenize(self, sentences, truncation=False):
        '''
        Given sentences, returns list of words with special tokens
        '''
        token_list = list()
        for sentence in sentences:
            arr = [self.cls_token] + [s for s in sentence.split(' ')]
            if self.max_len:
                if len(arr) < self.max_len:
                    pads =  [self.pad_token] * (self.max_len - len(arr) - 1)
                    arr = arr + pads
                else:
                    # Truncate
                    arr = arr[: self.max_len -1]
            arr = arr + [self.sep_token]
            token_list.append(arr)
        return token_list