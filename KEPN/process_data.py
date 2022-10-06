# -*- coding:utf-8 -*-
import os
from scripts.utils import create_vocab, load_vocab, save_vocab

print(os.path.abspath(os.path.curdir))
def generate_dict(dict_fpath, vocab_fpath, in_fpath, out_fpath, max_paraph_dict=15):
    '''
    dict_fpath：synonym file
    vocab_fpath：vocab used in our model
    '''
    vocab = load_vocab(vocab_fpath)[0].keys()
    vocab = set(vocab)
    synonym_dict = {}
    with open(dict_fpath, 'r', encoding="utf8") as f1:
        for line in f1:
            items = line.strip().split()
            if items[0][-1] != '=':
                continue
            items = items[1:]
            items = [item for item in items if item in vocab]
            for word1 in items:
                synonym_dict[word1] = items

    sents1, sents2 = [], []
    with open(in_fpath, 'r', encoding="utf8") as f1:
        for sent1 in f1:
            words = sent1.strip().split()
            sents1.append(words)
    print("size", len(sents1))

    paraphrase_pair = []
    synonym_count = 0
    for sent in sents1:
        sent_paraphrase = []
        word_paraphrase_record = set()
        s_count = 0
        for pos, word in enumerate(sent):
            if word not in synonym_dict: continue
            s_count += 1
            t = []
            word_paraphrase_record.add(word)
            for p_word in synonym_dict[word]:
                if p_word == word: continue
                t.append(p_word + "->" + str(pos))
            sent_paraphrase.append(t)

        synonym_count += s_count
        if len(sent_paraphrase) == 0:
            sent_paraphrase.append(["<UNK>-><UNK>"])
        count = 0
        f_result = []
        index = 0
        max_index = 10
        while count < max_paraph_dict and index < max_index:
            for line in range(len(sent_paraphrase)):
                if len(sent_paraphrase[line]) > max_index: max_index = len(sent_paraphrase[line])
                if index >= len(sent_paraphrase[line]): continue
                f_result.append(sent_paraphrase[line][index])
                count += 1
                if count >= max_paraph_dict:
                    break
            index += 1
        paraphrase_pair.append(f_result)

    max_index = -1
    with open(out_fpath, "w", encoding="utf8") as f:
        for line in paraphrase_pair:
            max_index = len(line) if max_index < len(line) else max_index
            if len(line) == 0:
                f.write("<UNK>-><UNK>\n")
                continue
            f.write(" ".join(line) + "\n")
    print("end!")
    return

dataset = 'mscoco'
vocab_num = 25000
file = [f'./dataset/{dataset}/train-src.txt',f'./dataset/{dataset}/train-tgt.txt']
save_path = f'./saved_model/{dataset}/vocab.pkl'
word2index, index2word = create_vocab(file_list=file,vocab_num=vocab_num)
save_vocab(word2index, index2word, save_path)

generate_dict("./dataset/Export_dict_synonym.txt", f"./saved_model/{dataset}/vocab.pkl", f"./dataset/{dataset}/train-src.txt", f"./dataset/{dataset}/train_paraphrased_pair.txt")
generate_dict("./dataset/Export_dict_synonym.txt", f"./saved_model/{dataset}/vocab.pkl", f"./dataset/{dataset}/val-src.txt", f"./dataset/{dataset}/val_paraphrased_pair.txt")
generate_dict("./dataset/Export_dict_synonym.txt", f"./saved_model/{dataset}/vocab.pkl", f"./dataset/{dataset}/test-src.txt", f"./dataset/{dataset}/test_paraphrased_pair.txt")