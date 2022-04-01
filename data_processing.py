# coding: utf-8

from collections import defaultdict
import operator

class Vocab(object):
    unk = u'<unk>'
    sos = u'<sos>'
    eos = u'<eos>'
    def __init__(self, unk=unk):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)
        # self.add_word(self.pad, count=0)
        # self.add_word(self.sos, count=0)
        # self.add_word(self.eos, count=0)

    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print('{} total words with {} uniques'.format(self.total_words, len(self.word_freq)))
 

    def limit_vocab_length(self, length):
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'w', encoding='utf8') as fd:
            for (word, freq) in sorted_tup:
                fd.write((u'%s\t%d\n'%(word, freq)))

    def save_vocab_for_nmt(self, filePath):
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'w', encoding='utf8') as fd:
            fd.write('<pad>\n')
            fd.write('<unk>\n')
            fd.write('<s>\n')
            fd.write('</s>\n')
            for (word, freq) in sorted_tup:
                fd.write((u'%s\n' % (word)))

    def load_vocab_from_file(self, filePath, sep='\t'):
        with open(filePath, 'r', encoding='utf8') as fd:
            for line in fd:
                word, freq = line.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = int(freq)
            print('load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq)))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

vocab = Vocab()
def read_data(filename):
    with open(filename,'r', encoding='utf8') as fd:
        step=0
        for line in fd:
            step+=1
            line_uni = line
            if line_uni.isspace():
                continue
            for word in line_uni.strip().split():
                vocab.add_word(word)
    print(filename + ' end', step)



def read_data_en(filename):
    from spacy.lang.en import English
    nlp = English()
    sentences = []
    with open(filename,'r', encoding='utf8') as fd:
        step=0
        for line in fd:
            step+=1
            line_uni = line
            if line_uni.isspace():
                continue
            line_uni = line_uni.strip()
            line_uni = line_uni.lower()
            doc = nlp(line_uni)
            sentences.append(' '.join([token.text  for token in doc]))      
    with open(file=filename + '.cp',mode='w',encoding='utf-8') as f:
        for s in sentences:
            f.write(s + '\n')
             
    print(filename + '.cp','read_data_en end', step)

dir_path = "./quora"


# read_data_en("%s/question1_train.src" % dir_path)
# read_data_en("%s/question2_train.tgt" % dir_path)
# read_data_en("%s/question1_dev.src" % dir_path)
# read_data_en("%s/question2_dev.tgt" % dir_path)
# read_data_en("%s/question1_test.src" % dir_path)
# read_data_en("%s/question2_test.tgt" % dir_path)

read_data("%s/quora.train.src.txt" % dir_path)
read_data("%s/quora.train.tgt.txt" % dir_path)

read_data("%s/quora.test.src.txt" % dir_path)
read_data("%s/quora.test.tgt.txt" % dir_path)

vocab.save_vocab('%s/quora.vocab.gaixie' % dir_path)
vocab = Vocab()
vocab.load_vocab_from_file('%s/quora.vocab.gaixie' % dir_path)
vocab.limit_vocab_length(120000)
vocab.save_vocab_for_nmt('%s/quora.vocab.txt' % dir_path)
