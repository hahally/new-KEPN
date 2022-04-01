import numpy as np
import tensorflow as tf
import warnings
import os
import json

warnings.filterwarnings('ignore')
import random

random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)

def read_file(file):
    lines = []
    with open(file=file,mode='r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            lines.append(line)
            line = f.readline()
            
    return lines


def load_data(fpath1, fpath2, paraphrased_fpath, maxlen1, maxlen2):
    sents1, sents2, paraphrased_pairs = [], [], []
    s1,s2,s3 = [read_file(file) for file in [fpath1, fpath2, paraphrased_fpath]]
    L1,L2,L3 = [len(s) for s in [s1,s2,s3]]
    assert len(s1)==len(s2)==len(s3), f'Please check the number of rows: {fpath1}:{L1},{fpath2}:{L2},{paraphrased_fpath}:{L3}'
    for sent1, sent2, dict_pair in zip(s1,s2,s3):
        if not sent1.strip(): continue
        if not sent2.strip(): continue
        if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
        if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
        sents1.append(sent1.strip())
        sents2.append(sent2.strip())
        paraphrased_pairs.append(dict_pair.strip())
        
    return sents1, sents2, paraphrased_pairs


def load_vocab(file):
    word2idx = {}
    idx2word = {}
    with open(file=file,mode='r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            word = line.split()[0]
            if word not in word2idx.keys():
                word2idx[word] = len(word2idx)
                idx2word[len(idx2word)] = word
            line = f.readline().strip()
            
    return word2idx, idx2word


def generator_fn(sents1, sents2, paraphrased_pairs, vocab_fpath, paraphrase_type=1):
    word2idx, idx2word = load_vocab(vocab_fpath)
    for sent1, sent2, parap_pair in zip(sents1, sents2, paraphrased_pairs):
        sent1, sent2 = sent1.decode("utf-8"), sent2.decode("utf-8")
        
        input_words = ['<pad>'] + sent1.split() #  + ['</s>']
        x = [word2idx.get(t, word2idx["<unk>"]) for t in input_words]
        y = [word2idx.get(t, word2idx["<unk>"]) for t in ["<s>"] + sent2.split() + ["</s>"]]
        # decoder_input, y = y[:-1], y[1:]
        x_paraphrased_dict = []
        synonym_label = []
        word_set,pos_set = set(),set()
        
        parap_pair = parap_pair.decode("utf-8")
        for word_pos in parap_pair.split():
            # word_pos: word->pos
            word, pos = word_pos.split('->')
            # paraphrase_type =0: x_paraphrased_dict:[[word1,word2],...], word1 是句子中的词，word2 是 word1对应的同义词
            # paraphrase_type =1: x_paraphrased_dict:[[word,pos],...]
            
            if pos == '<unk>': pos = -1
            if word == '<unk>': word = input_words[0]
            
            word1 = input_words[int(pos)+1]
            word2 = word
            if paraphrase_type==0:
                word_set.add(word1)
                x_paraphrased_dict.append([word2idx.get(word1, word2idx['<unk>']), word2idx.get(word2, word2idx['<unk>'])])
            
            if paraphrase_type==1:
                pos_set.add(int(pos)+1)
                x_paraphrased_dict.append([word2idx.get(word2, word2idx['<unk>']), int(pos)+1])
            
        synonym_label = [int(i in pos_set) if paraphrase_type else int(w in word_set) for i, w in enumerate(input_words)]
        src = x
        tgt = y
        
        # yield src,sent1, tgt,sent2, x_paraphrased_dict, synonym_label
        yield src, tgt, x_paraphrased_dict, synonym_label


def get_dataset(sents1, sents2, paraphrased_pairs, vocab_fpath, batch_size, shuffle=False, paraphrase_type=1):
    output_types = (tf.int32, tf.int32, tf.int32, tf.int32)
    output_shapes = ((None,),
                     (None,),
                     (None,2),
                     (None,)
                     )
    dataset = tf.data.Dataset.from_generator(generator_fn, 
                                              args=(sents1, sents2, paraphrased_pairs, vocab_fpath, paraphrase_type), 
                                              output_types= output_types, 
                                              output_shapes = output_shapes
                                              )
    if shuffle:
        dataset = dataset.shuffle(10000*batch_size)
        
    # dataset = dataset.repeat()  # 这行有毒
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=output_shapes, padding_values=(0,0,0,2))
    dataset = dataset.prefetch(1)
    
    return dataset

def save_hparams(hparams, path):
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)
   

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


if __name__=='__main__':
    
    # word2idx, idx2word = load_vocab('./data/question.vocab')
    sents1, sents2, paraphrased_pairs = load_data('quora/quora.train.src.txt','quora/quora.train.tgt.txt','quora/train_paraphrased_pair.txt',50,50)
    # gen = generator_fn(sents1, sents2, paraphrased_pairs, './data/question.vocab', paraphrase_type=1)
    # # print(len(sents1),len(sents2),len(paraphrased_pairs))
    # for g in gen:
    #     print(g)
    fpath1 = './data/question1_dev.src.cp'
    fpath2 = './data/question2_dev.tgt.cp'
    maxlen1 = 50
    maxlen2 = 50
    vocab_fpath = './data/question.vocab'
    paraphrased_fpath = './data/dev_paraphrased_pair.txt'
    batch_size = 8
    from tqdm import tqdm
    dt = get_dataset(sents1[:64*10], sents2[:64*10], paraphrased_pairs[:64*10],'quora/quora.vocab.txt',2,shuffle=True,paraphrase_type=1)
    for batch, inputs in tqdm(enumerate(dt)):
        print(inputs)
        break
    # print(inputs)
    print('end')
