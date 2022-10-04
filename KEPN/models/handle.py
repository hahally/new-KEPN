from tqdm import tqdm
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

from scripts.Constants import EOS

def pad_mask(inputs, PAD):
    return (inputs==PAD).unsqueeze(1)


def triu_mask(length):
    mask = torch.ones(length, length).triu(1)
    return mask.unsqueeze(0) == 1

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def get_pos_emb(self, x):
        pos_emb = self.pe[:, : x.size(1)].requires_grad_(False)
        
        return pos_emb

    def forward(self, x):
        x = x + self.get_pos_emb(x)
        
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=6,
                 d_ff=1024,
                 dropout=0.1,
                 d_model=512):
        super(TransformerModel, self).__init__()
        self.nhead = nhead
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        self.positional_encoding = PositionalEncoding(d_model, dropout=0.3)
        
    def generate_square_subsequent_mask(self, sz: int):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt):
        # 生成 mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.size()[-1]).to(src.device)
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt)

        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        # 给src和tgt的token增加位置信息
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        # torch1.8.1 版本要求 src_emb，tgt_emb： seq_len,N,d_model
        # src_emb.transpose(1,0,2)
        memory = self.encoder(
            src_emb.transpose(1, 0), src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb.transpose(1, 0), memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)

        return output.transpose(1, 0), memory.transpose(1, 0)

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        
        return tokens == 0
    
class SoftAttention(nn.Module):
    def __init__(self,d_model=512):
        super(SoftAttention, self).__init__()
        self.d_model = d_model
        
        self.si_w = nn.Sequential(nn.Linear(self.d_model*2, self.d_model*2), nn.Tanh())
        self.si_v = nn.Sequential(nn.Linear(self.d_model*2, self.d_model*2), nn.Softmax(dim=-1))
        self.pi_w = nn.Sequential(nn.Linear(self.d_model*2, self.d_model*2), nn.Tanh())
        self.pi_v = nn.Sequential(nn.Linear(self.d_model*2, self.d_model*2), nn.Softmax(dim=-1))
    
    def forward(self, syn_emb, pos_emb, dec):
        paraphrased_lens = syn_emb.shape[1]
        seq_lens = dec.shape[1]
        batch = syn_emb.shape[0]
        
        y_logit_ = torch.unsqueeze(dec, dim=2) * torch.ones(batch,seq_lens,paraphrased_lens,self.d_model,device=dec.device) # b,seqlen,para_len,dim
        syn_emb_ = torch.unsqueeze(syn_emb, dim=1) * torch.ones(batch,seq_lens,paraphrased_lens,self.d_model,device=syn_emb.device) # b,seqlen,para_len,dim
        pos_emb_ = torch.unsqueeze(pos_emb, dim=1) * torch.ones(batch,seq_lens,paraphrased_lens,self.d_model,device=syn_emb.device) # b,seqlen,para_len,dim
        
        h_si_concat = torch.cat([y_logit_, syn_emb_], -1)
        h_pi_concat = torch.cat([y_logit_, pos_emb_], -1)
        
        att_si = self.si_v(self.si_w(h_si_concat))
        att_pi = self.pi_v(self.pi_w(h_pi_concat))
        
        si_score = torch.sum(att_si,dim=-1) # b,seq_len,para_len
        pi_score = torch.sum(att_pi,dim=-1)
        
        ct_si = torch.matmul(si_score, syn_emb) # b,seq_len,d
        ct_pi = torch.matmul(pi_score, pos_emb)
        
        ct = torch.cat([ct_si, ct_pi], dim=-1) # b,seq_len,d*2
        
        return ct

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0., ignore_index=None):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, norm=1):
        inputs = torch.log(inputs)
        vocab_size = inputs.size(-1)
        batch_size = targets.size(0)
        length = targets.size(1)
        if self.ignore_index is not None:
            mask = (targets == self.ignore_index).view(-1)
        
        index = targets.unsqueeze(-1)
        targets = F.one_hot(targets, num_classes=inputs.size(-1))
        targets = targets * (1 - self.smoothing) + self.smoothing / vocab_size
        loss = self.criterion(inputs.view(-1, vocab_size), 
                              targets.view(-1, vocab_size).detach()).sum(dim=-1)
        if self.ignore_index is not None:
            return loss.masked_fill(mask, 0.).sum() / norm
        else:
            return loss.sum() / norm
        
# TO DO
class Generator():
    def __init__(self, idx2word, model, max_len=30):
        super(Generator).__init__()
        self.id2word = idx2word
        self.model = model
        self.max_len = max_len
    
    def generate(self, dataloader):
        sents = []
        self.model.eval()
        with torch.no_grad():
            for src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, synonym_label in tqdm(dataloader):
                tgt_tokens = self.generate_batch(src_tokens, syn_tokens, pos)
                sents += self.idx2sent(tgt_tokens)
        
        return sents
        
    def generate_batch(self,src, syn, pos):
        bsz = src.size(0)
        tgt = torch.zeros(size=(bsz, 1), dtype=torch.long).to(src.device)
        for i in range(self.max_len):
            _, out = self.model(src, tgt,syn, pos)
            
            word_pre = out[:,-1]
            word_index = torch.argmax(word_pre, dim=1)
            tgt = torch.cat([tgt, word_index.unsqueeze(1)], dim=1)
            
            if word_index == EOS:
                break
            
        return tgt
    
    def idx2sent(self, sent_tokens):
        sents = []
        for tokens in sent_tokens:
            sents.append(' '.join([self.id2word[int(idx)] for idx in tokens]))
            
        return sents