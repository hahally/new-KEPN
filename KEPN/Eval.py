from bert_score import score
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu

class Eval:
    def __init__(self, source_file, reference_file):
        self.source = self.read_file(source_file, reference=False)
        self.source_ref = self.read_file(source_file, reference=True)
        self.reference = self.read_file(reference_file, reference=True)

    def read_file(self, file, reference=False):
        with open(file, 'r', encoding='UTF-8') as f:
            if reference:
                data = [[[word.lower() for word in seq.strip('\n').split()] 
                          for seq in line.strip('\n').split('\t')] for line in f.readlines()]
            else:
                data = [[word.lower() for word in line.strip('\n').split()] for line in f.readlines()]
                
        return data
    
    def meteor(self, reference, candidate):
        reference = [[' '.join(r) for r in ref] for ref in reference]
        candidate = [' '.join(cand) for cand in candidate]
        ms = 0
        for r, c in zip(reference, candidate):
            ms += meteor_score(r, c)
            
        avg_ms = round(ms/len(candidate)*100, 2)
        return avg_ms
    
    def bleu(self, reference, candidate):
        bleu4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        bleu4 = round(bleu4*100, 2)
        return bleu4
    
    def bertscore(self, reference, candidate):
        reference = [[' '.join(ref)] for ref in reference]
        candidate = [' '.join(cand) for cand in candidate]
        model_type='bert-base-uncased'
        (P, R, F), hashname = score(candidate, reference, model_type=model_type, batch_size=128,lang="en", return_hash=True, rescale_with_baseline=True)
        
        bs = round(F.mean().item()*100, 2)
        
        return bs
    
    def __call__(self, candidate_file):
        candidate = self.read_file(candidate_file, reference=False)
        bleu = self.bleu(self.reference, candidate)
        self_bleu = self.bleu(self.source_ref, candidate)
        bertScore = self.bertscore(self.source, candidate)
        meteor = self.meteor(self.reference, candidate)
        result = {}
        result['self-BLEU'] = self_bleu
        result['BLEU'] = bleu
        result['iBLEU'] = round(0.7*bleu - 0.3*self_bleu, 2)
        result['bert-score'] = bertScore
        result['meteor'] = meteor
        
        return result

# 保存文件
def save_text(lines,file):
    with open(file=file, mode='w', encoding='utf-8') as f:
        for line in lines:
            f.write(line+'\n')

if __name__ == '__main__':
    dataset = 'mscoco'
    source_file = f'./dataset/{dataset}/test-src.txt'
    reference_file = f'./dataset/{dataset}/test-tgt.txt'
    save_path = f'./saved_model/{dataset}/{dataset}-score.txt'
    lines = []
    eval = Eval(source_file=source_file, reference_file=reference_file)
    cnad_file = f'./saved_model/{dataset}/test.txt'
    re = eval(candidate_file=cnad_file)
    print(f"{cnad_file}:{re}")
    line_info = ''
    for k,v in re.items():
        line_info += f'\t{k}:{v}'
    line = f"{cnad_file}:{line_info}"
    lines.append(line)
        
    save_text(lines=lines, file=save_path)