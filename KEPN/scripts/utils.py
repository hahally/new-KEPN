import os
import pickle

import torch
from scripts.Constants import *

def create_vocab(file_list, vocab_num=-1):
    def create_corpus(file):
        with open(file, 'r', encoding='utf-8') as f:
            corpus = [word.lower() for line in f.readlines() for word in line.strip('\n').split()]
        return corpus
    corpus = []
    for file in file_list:
        corpus.extend(create_corpus(file))
    
    word2index = {}; index2word = {}
    word2index[PAD_WORD] = PAD; index2word[PAD] = PAD_WORD
    word2index[UNK_WORD] = UNK; index2word[UNK] = UNK_WORD
    word2index[BOS_WORD] = BOS; index2word[BOS] = BOS_WORD
    word2index[EOS_WORD] = EOS; index2word[EOS] = EOS_WORD
    if vocab_num != -1:
        word_count = {}
        for word in corpus:
            if word_count.get(word) is None:
                word_count[word] = 1
            else:
                word_count[word] += 1
        w_count = [[word, word_count[word]] for word in word_count.keys()]
        w_count.sort(key=lambda elem: elem[1], reverse=True)
        w_count = [w_count[i][0] for i in range(min(len(w_count), vocab_num))]
    else:
        w_count = set(corpus)
    for word in w_count:
        word2index[word] = len(word2index)
        index2word[len(index2word)] = word
        
    return word2index, index2word

def save_model(model, save_path):
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path))
    
    return model

# 保存文件
def save_text(lines,file):
    with open(file=file, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')

def load_vocab(save_path):
    with open(save_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab['word2idx'], vocab['idx2word']

def save_vocab(word2index, index2word, save_path):
    vocab = {'word2idx':  word2index,
             'idx2word':  index2word}
    
    file_path = os.path.join(*os.path.split(save_path)[:-1])
    if os.path.exists(file_path) == False:
        os.makedirs(file_path)
    
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)
    print('===> Save Vocabulary Successfully.')