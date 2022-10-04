
from importlib.resources import path
from tqdm import tqdm
import torch
from models.handle import LabelSmoothing
from scripts.Constants import PAD
from scripts.data_load import create_dataloader
from scripts.utils import load_vocab, save_model
from models.model import KEPN

def train_model(model, device, data, epoch, batchPrintInfo, max_length):
    train_loader = data['train']
    valid_loader = data['valid']
    # model = KEPN(vocab_size=1000)
    alpha = 0.1
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=0.0001,
                             betas=(0.9, 0.98),
                             eps=1e-9,
                             weight_decay=1e-4)
    criterion_labeling = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothing(smoothing=0.1, ignore_index=PAD).to(device)
    
    for e in range(epoch):
        model.train()
        acc_label = 0
        loss_total = 0
        for i, (src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, synonym_label) in enumerate(train_loader):
            optim.zero_grad()
            src_tokens = src_tokens.to(device)
            tgt_sent_in = tgt_sent_in.to(device)
            tgt_sent_out = tgt_sent_out.to(device)
            syn_tokens = syn_tokens.to(device)
            pos = pos.to(device)
            synonym_label = synonym_label.to(device)
            
            prediction, output = model(src_tokens, tgt_sent_in, syn_tokens, pos)
            
            bsz, src_len = synonym_label.shape
            prediction = prediction.reshape(bsz * src_len, 2)
            synonym_label = synonym_label.reshape(bsz * src_len)
            loss_labeling = criterion_labeling(prediction,synonym_label).to(device)
            acc = torch.sum((torch.argmax(prediction, dim=-1) == synonym_label)).item()/(bsz * src_len)
            acc_label += acc
            loss_g = criterion(output, tgt_sent_out).to(device)
            
            loss = alpha * loss_labeling + (1-alpha) * loss_g
            loss_total += loss.item()
            loss.backward()
            optim.step()
            
            # print
            if (i+1)%batchPrintInfo==0:
                log_info = f'Epoch[{e+1}], step[{i+1}], loss: {loss.item()}, loss_g: {loss_g.item(), loss_labeling: {loss_labeling.item()}}'
                print(log_info)
        
        # 保存模型
        path = f'last-{(e+1)%5}-model.pth'
        save_model(model, path)
        
        # 评估
        acc = 0
        loss_total = 0
        loss_labeling = 0
        loss_g = 0
        model.eval()
        with torch.no_grad():
            for (src_tokens, tgt_sent_in, tgt_sent_out, syn_tokens, pos, synonym_label) in tqdm(valid_loader):
                src_tokens = src_tokens.to(device)
                tgt_sent_in = tgt_sent_in.to(device)
                tgt_sent_out = tgt_sent_out.to(device)
                syn_tokens = syn_tokens.to(device)
                pos = pos.to(device)
                synonym_label = synonym_label.to(device)
                
                prediction, output = model(src_tokens, tgt_sent_in, syn_tokens, pos)
                
                bsz, src_len = synonym_label.shape
                prediction = prediction.reshape(bsz * src_len, 2)
                synonym_label = synonym_label.reshape(bsz * src_len)
                loss_labeling += criterion_labeling(prediction,synonym_label).to(device)
                acc += torch.sum((torch.argmax(prediction, dim=-1) == synonym_label)).item()/(bsz * src_len)
                
                loss_g += criterion(output, tgt_sent_out).to(device)
                loss = alpha * loss_labeling + (1-alpha) * loss_g
                loss_total += loss.item()
            
            log_info = f'Epoch[{e+1}], Eval, loss: {loss.item()}, loss_g: {loss_g.item(), loss_labeling: {loss_labeling.item()}}'
            print(log_info)
        
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
    data = {}
    data['train'] = dataloader
    device = device = torch.device('cpu')
    model = KEPN(vocab_size=100).to(device)
    
    train_model(model, device, data, epoch=1, batchPrintInfo=100, max_length=25)