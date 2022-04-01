import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32100, type=int)

    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    
    
    
    # train
    ## files
    parser.add_argument('--train_src', default='quora/quora.train.src.txt')
    parser.add_argument('--train_tgt', default='quora/quora.train.tgt.txt')
    parser.add_argument('--train_paraphrased', default='quora/train_paraphrased_pair.txt')

    parser.add_argument('--test_src', default='quora/quora.test.src.txt')
    parser.add_argument('--test_tgt', default='quora/quora.test.tgt.txt')
    parser.add_argument('--test_paraphrased', default='quora/test_paraphrased_pair.txt')

    parser.add_argument('--paraphrase_type', default=1, type=int)

    ## vocabulary
    parser.add_argument('--vocab_path', default='quora/quora.vocab.txt', help="vocabulary file path")

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
    parser.add_argument('--l_alpha', default=0.9, type=float, help="the weighting coefficient for trade-off between loss1 and loss2.")
    parser.add_argument('--num_epochs', default=10, type=int)


    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    
    parser.add_argument('--maxlen1', default=50, type=int,
                        help="maximum length of a source sequence")
    
    parser.add_argument('--maxlen2', default=50, type=int,
                        help="maximum length of a target sequence")
    
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    parser.add_argument('--ckpt', default="checkpoints/train", help="checkpoint file path")
    parser.add_argument('--logdir', default="log/")