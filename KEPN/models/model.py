import torch
import torch.nn as nn
from models.module import TransformerModel,SoftAttention

class KEPN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.3,
                 d_model=300):
        super(KEPN, self).__init__()
        
        self.transformer_model = TransformerModel(vocab_size=vocab_size,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers,
                                                  d_ff=d_ff,
                                                  dropout=dropout,
                                                  d_model=d_model)

        self.soft_att = SoftAttention(d_model=d_model)

        self.mlp = nn.Sequential(nn.Linear(d_model*3, vocab_size))
        self.labeling = nn.Sequential(nn.Linear(d_model, 3))
    
    def forward(self, src, tgt, syn, pos):
        
        dec, memory = self.transformer_model(src, tgt)
        prediction = self.labeling(memory)
        
        syn_emb = self.transformer_model.embedding(syn) # b,syn_len,d_model
        pos_emb = self.transformer_model.positional_encoding.pe[0, pos] # b,pos_len,d_model
        ct = self.soft_att(syn_emb,pos_emb) # b,seq_len,d_model*2
        output = self.mlp(torch.cat([dec, ct], dim=-1)) # b,se1_len,vocab_size
    
        return prediction, output
