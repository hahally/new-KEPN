
from scripts.data_load import create_dataloader
from scripts.utils import load_vocab
from models.model import KEPN

def train_model(model, criterion, optim, device, data, epoch, batchPrintInfo, max_length):
    train_loader = data['train']
    valid_loader = data['valid']
    model = KEPN(vocab_size=1000)
    for i, (src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, label) in enumerate(dataloader):
        src_tokens = src_tokens.to(device)
        tgt_sent_in = tgt_sent_in.to(device)
        tgt_sent_out = tgt_sent_out.to(device)
        syn_tokens = syn_tokens.to(device)
        pos = pos.to(device)
        label = label.to(device)
        
        prediction, output = model(src_tokens, tgt_sent_in, syn_tokens, pos)
        



if __name__ == '__main__':
    dataset = 'quora'
    vocab_path = f'./saved_model/{dataset}/vocab.pkl'
    word2index, index2word = load_vocab(vocab_path)
    dataloader = create_dataloader(tokenizer=word2index,
                             src_file='./dataset/quora/train-src.txt',
                             tgt_file='./dataset/quora/train-tgt.txt',
                             pair_file='./dataset/quora/train_paraphrased_pair.txt',
                             batch_size=2,
                             shuffle=False)
    
    for i, (src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, label) in enumerate(dataloader):
        model = KEPN(vocab_size=100)
        prediction, output = model(src_tokens, tgt_sent_in, syn_tokens, pos)
        print(src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, label)