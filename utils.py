# coding: UTF-8
import os
import pandas as pd
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import struct
import binascii


MAX_VOCAB_SIZE = 100000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def get_num_binary(num):
    # 1. 将浮点数打包到4字节binary string
    packed = struct.pack('!f', num)
    # print('Packed:', packed)

    # 2. 用hexlify()转为16进制表示
    hex_data = binascii.hexlify(packed)
    # print('Hex:', hex_data)

    # 3. 16进制转为32位二进制字符串
    binary = bin(int(hex_data, 16))[2:].zfill(32)
    # print('Binary:', binary)

    # 4. 打印32位二进制
    bits = [int(_x) for _x in binary]
    # for bit in binary:
    #     print(bit, end='')
    return bits


def bit_embedding(arr):
    res = []
    _batch = len(arr)
    for num in arr:
        res.append(get_num_binary(num))
    return res


def build_vocab(file_path, max_size, min_freq, use_word=False):
    """
    统计词汇，建立词汇表
    """
    vocab_dic = {}
    tokenizer = lambda x: [y for y in x]
    df = pd.read_csv(file_path)

    for idx, row in tqdm(df.iterrows()):
        if use_word:
            content = row['word_level_improved']
            tokenizer = lambda x: x.split(' ')      # 以空格隔开，word-level
        else:
            content = row['url']
            tokenizer = lambda x: [y for y in x]    # char-level
        content = f'{content}'.strip()
        if not content:
            continue
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, use_word):
    if use_word:
        level = 'word_level_improved'
        tokenizer = lambda x: x.split(' ')
    else:
        level = 'url'
        tokenizer = lambda x: [y for y in x]

    print('URL Preprocessing Mode: ', level)

    # 加载或建立词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, max_size=MAX_VOCAB_SIZE, min_freq=1, use_word=use_word)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))

    config.n_vocab = len(vocab)
    print(f"Vocab size: {config.n_vocab}")

    def load_dataset(path, pad_size=128):
        contents = []
        df = pd.read_csv(path, encoding='ISO-8859-1')
        for idx, row in tqdm(df.iterrows()):
            content = f'{row[level]}'.strip()
            lexical = row.iloc[1:-1].values
            label = row['label']
            words_line = []
            if not content:
                continue
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if seq_len < pad_size:
                    token.extend([PAD] * (pad_size-seq_len))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, bit_embedding(lexical), int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = len(batches) % batch_size != 0  # 记录batch数量是否为整数
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        l = torch.FloatTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[-1] for _ in datas]).to(self.device)
        return (x, l), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):

    iter = DatasetIterater(dataset, config.batch_size, config.device)

    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    build_vocab(file_path='ISCXURL2016/train.csv', max_size=MAX_VOCAB_SIZE, min_freq=1, use_word=True)


