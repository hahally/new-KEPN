import torch
from models.handle import Generator
from models.model import KEPN
from scripts.data_load import create_dataloader
from scripts.utils import load_model, load_vocab, save_text

dataset = 'mscoco'
vocab_path = f'./saved_model/{dataset}/vocab.pkl'
word2index, index2word = load_vocab(vocab_path)
checkpoint_path = './saved_model/quora/last-2-model.pth'
save_path = f'./saved_model/{dataset}/test.txt'
test_loader = create_dataloader(tokenizer=word2index,
                          src_file=f'./dataset/{dataset}/test-src.txt',
                          tgt_file=f'./dataset/{dataset}/test-tgt.txt',
                          pair_file=f'./dataset/{dataset}/test_paraphrased_pair.txt',
                          batch_size=100,
                          shuffle=False)

device = torch.device('cuda:0')
model = KEPN(vocab_size=len(word2index),
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=8,
                 d_ff=2048,
                 dropout=0.3,
                 d_model=512).to(device)
model = load_model(model, checkpoint_path)

G = Generator(idx2word=index2word, model=model, device=device)
sents = G.generate(dataloader=test_loader)

save_text(sents, save_path)