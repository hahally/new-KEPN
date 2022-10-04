import random
import torch
from torch.utils.data import Dataset,DataLoader
from scripts.Constants import *

class KEPNDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 src_file,
                 tgt_file,
                 pair_file,
                 clip_len = 20):
        super(KEPNDataset).__init__()
            
        self.src = self.read_file(src_file)
        self.tgt = self.read_file(tgt_file)
        self.tokenizer = tokenizer
        self.clip_len = clip_len
        
        self.syn, self.pos = self.get_syn_pos_pair(pair_file)
    
    def clip_sent(self, sent):
        if len(sent)>self.clip_len:
            ed = sent[-1]
            sent = sent[:self.clip_len-1]
            sent.append(ed)
        
        return sent
    
    def __getitem__(self, index):
        src_sent = self.clip_sent(self.src[index])
        tgt_sent = self.clip_sent(self.tgt[index])
        
        s = self.syn[index]
        p = torch.LongTensor(self.pos[index])
        
        src_sent = src_sent + [PAD_WORD]*(self.clip_len - len(src_sent))
        
        tgt_sent_in = [BOS_WORD] + tgt_sent + [PAD_WORD]*(self.clip_len - len(tgt_sent))
        tgt_sent_out = tgt_sent + [EOS_WORD] + [PAD_WORD]*(self.clip_len - len(tgt_sent))
        src_tokens = torch.LongTensor([self.tokenizer.get(w, UNK) for w in src_sent])
        tgt_sent_in = torch.LongTensor([self.tokenizer.get(w, UNK) for w in tgt_sent_in])
        tgt_sent_out = torch.LongTensor([self.tokenizer.get(w, UNK) for w in tgt_sent_out])
        syn_tokens = torch.LongTensor([self.tokenizer.get(w, UNK) for w in s])
        
        label = torch.LongTensor([int(i in p) for i, w in enumerate(src_sent)])
        
        return src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, p, label
        
    def read_file(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            sents = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
            
        return sents
    
    def get_syn_pos_pair(self, file):
        syn = []
        pos = []
        with open(file, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            while line:
                pairs = line.split()
                if len(pairs)>self.clip_len:
                    pairs = random.sample(pairs, self.clip_len)
                pairs = pairs + [f'{PAD_WORD}->30'] * (self.clip_len - len(pairs))
                tmp_syn = []
                tmp_pos = []
                for pair in pairs:
                    s, p = pair.split('->')
                    if p == UNK_WORD:
                        p = 30
                        s = PAD_WORD
                    tmp_syn.append(s)
                    tmp_pos.append(int(p))
                syn.append(tmp_syn)
                pos.append(tmp_pos)
                line = f.readline().strip()
        
        return syn, pos
    
    def __len__(self):
        
        return len(self.src)
    

def create_dataloader(tokenizer,
                      src_file,
                      tgt_file,
                      pair_file,
                      clip_len=20,
                      batch_size=1,
                      shuffle=True,
                      num_workers=0,
                      pin_memory=False):

    dataset = KEPNDataset(tokenizer, src_file, tgt_file, pair_file, clip_len)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    
    return dataloader
    
