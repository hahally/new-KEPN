from tensorflow.python.keras.engine import training
from models import *
from utils import *
import os
import tensorflow as tf

import random
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from hparams import Hparams

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

# 参数
train_src = hp.train_src
train_tgt = hp.train_tgt
train_paraphrased = hp.train_paraphrased

test_src = hp.test_src
test_tgt = hp.test_tgt
test_paraphrased = hp.test_paraphrased

vocab_path = hp.vocab_path

batch_size = hp.batch_size
num_epochs = hp.num_epochs
shuffle = True
print_freq = hp.print_freq
save_freq = hp.save_freq
lr = hp.lr
l_alpha = hp.l_alpha

checkpoint_path = hp.ckpt

seed = hp.seed

# 模型参数
maxlen1 = hp.maxlen1
maxlen2 = hp.maxlen2
vocab_size = hp.vocab_size
d_model = hp.d_model
num_heads = hp.num_heads
ff_dim = hp.d_ff
dropout_rate = hp.dropout_rate
num_blocks = hp.num_blocks
paraphrase_type = hp.paraphrase_type

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

logging.info("# load data")
sents1, sents2, paraphrased_pairs = load_data(train_src, train_tgt, train_paraphrased, maxlen1, maxlen2)
train_dataset = get_dataset(sents1, sents2, paraphrased_pairs, vocab_path, batch_size, shuffle, paraphrase_type)

sents1, sents2, paraphrased_pairs = load_data(test_src, test_tgt, test_paraphrased, maxlen1, maxlen2)
val_dataset = get_dataset(sents1, sents2, paraphrased_pairs, vocab_path, batch_size, shuffle=False, paraphrase_type = paraphrase_type)

model = Transformer(maxlen1, vocab_size, d_model, num_heads, ff_dim, dropout_rate, num_blocks, paraphrase_type)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred, mode = 'generate'):
    assert mode in ['generate','labeling']
    if mode == 'generate':
        padding_values = 0
    
    if mode == 'labeling':
        padding_values = 2
    mask = tf.math.logical_not(tf.math.equal(real, padding_values))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred, mode = 'generate'):
    assert mode in ['generate','labeling']
    if mode == 'generate':
        padding_values = 0
    if mode == 'labeling':
        padding_values = 2
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, padding_values))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
labeling_accuracy = tf.keras.metrics.Mean(name='labeling_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
val_labeling_accuracy = tf.keras.metrics.Mean(name='val_labeling_accuracy')

ckpt = tf.train.Checkpoint(model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

def train_step(encoder_inputs, y, x_paraphrased_dict, synonym_label):
    decoder_input,y = y[:,:-1], y[:,1:]
    with tf.GradientTape() as tape:
        predictions, synonym_labeling_out = model((encoder_inputs, decoder_input, x_paraphrased_dict))
        loss1 = loss_function(y, predictions, mode='generate')
        loss2 = loss_function(synonym_label, synonym_labeling_out, mode='labeling')
        total_loss = l_alpha*loss1 + (1-l_alpha)*loss2

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(total_loss)
    train_accuracy(accuracy_function(y, predictions, mode='generate'))
    labeling_accuracy(accuracy_function(synonym_label, synonym_labeling_out, mode='labeling'))

def eval(encoder_input, y, x_paraphrased_dict, synonym_label):
    decoder_input,y = y[:,:-1], y[:,1:]
    predictions, synonym_labeling_out = model((encoder_input, decoder_input, x_paraphrased_dict),training=False)
    loss1 = loss_function(y, predictions, mode='generate')
    loss2 = loss_function(synonym_label, synonym_labeling_out, mode='labeling')
    total_loss = l_alpha*loss1 + (1-l_alpha)*loss2
    val_loss(total_loss)
    val_accuracy(accuracy_function(y, predictions, mode='generate'))
    val_labeling_accuracy(accuracy_function(synonym_label, synonym_labeling_out, mode='labeling'))

logging.info("# start training...")

flag = -1
if os.path.exists('./log/flag.txt'):
    with open('flag.txt', mode='r') as f:
        flag = int(f.readlines()[0].strip())
             
for epoch in range(num_epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    labeling_accuracy.reset_states()
    
    val_loss.reset_states()
    val_accuracy.reset_states()
    val_labeling_accuracy.reset_states()
    for batch, inputs in enumerate(train_dataset):
        if batch > flag:
            src, tgt, x_paraphrased_dict, synonym_label = inputs
            encoder_inputs = tf.cast(src, dtype=tf.int64)
            decoder_input = tf.cast(tgt, dtype=tf.int64)
            x_paraphrased_dict = tf.cast(x_paraphrased_dict, dtype=tf.int64)
            synonym_label = tf.cast(synonym_label, dtype=tf.int64)
            train_step(encoder_inputs, decoder_input, x_paraphrased_dict, synonym_label)
            
            if (batch+1) % print_freq == 0:
                print(f'Epoch {epoch + 1} Batch {batch+1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Labeling_acc {labeling_accuracy.result():.4f}')

            if (batch+1) % save_freq == 0:
                print(f'Epoch {epoch + 1} Batch {batch+1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Labeling_acc {labeling_accuracy.result():.4f}')
                ckpt_save_path = ckpt_manager.save()
                print('saved checkpoint!')
                # 保存 batch 位置
                with open('./log/flag.txt', mode='w', encoding='utf-8') as f:
                    f.write(str(batch))
    
    for batch, inputs in enumerate(val_dataset):
        src, tgt, x_paraphrased_dict, synonym_label = inputs
        encoder_inputs = tf.cast(src, dtype=tf.int64)
        decoder_input = tf.cast(tgt, dtype=tf.int64)
        x_paraphrased_dict = tf.cast(x_paraphrased_dict, dtype=tf.int64)
        synonym_label = tf.cast(synonym_label, dtype=tf.int64)
        eval(encoder_inputs, decoder_input, x_paraphrased_dict, synonym_label)
        # print(f'Epoch {epoch + 1} Val Loss {val_loss.result():.4f} val_accuracy {val_accuracy.result():.4f} val_labeling_accuracy {val_labeling_accuracy.result():.4f}') 
    print(f'Epoch {epoch + 1} Train Loss {train_loss.result():.4f} train_accuracy {train_accuracy.result():.4f} labeling_accuracy {labeling_accuracy.result():.4f}') 
    print(f'Epoch {epoch + 1} Val Loss {val_loss.result():.4f} val_accuracy {val_accuracy.result():.4f} val_labeling_accuracy {val_labeling_accuracy.result():.4f}')    
    ckpt_save_path = ckpt_manager.save()
    print('saved checkpoint!')           




