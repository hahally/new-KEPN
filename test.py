
from models import *
from utils import *

import tensorflow as tf

import random
import numpy as np

random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


sents1, sents2, paraphrased_pairs = load_data('./quora/quora.test.src.txt','./quora/quora.test.tgt.txt','./quora/test_paraphrased_pair.txt',50,50)
test_dataset = get_dataset(sents1, sents1, paraphrased_pairs,'./quora/quora.vocab.txt',1,shuffle=True,paraphrase_type=1)

model = Transformer()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

vocab_fpath = './quora/quora.vocab.txt'
word2idx, idx2word = load_vocab(vocab_fpath)

src, tgt, x_paraphrased_dict, synonym_label = next(iter(test_dataset))

encoder_inputs = tf.cast(src, dtype=tf.int64)
x_paraphrased_dict = tf.cast(x_paraphrased_dict, dtype=tf.int64)
synonym_label = tf.cast(synonym_label, dtype=tf.int64)

start = tf.cast(2,dtype=tf.int64)[tf.newaxis]
end = tf.cast(3,dtype=tf.int64)[tf.newaxis]

output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
output_array = output_array.write(0, start)
for i in range(50):
    output = tf.transpose(output_array.stack())
    predictions, _ = model((encoder_inputs, output, x_paraphrased_dict), training=False)
    predictions = predictions[:, -1:, :]
    predicted_id = tf.argmax(predictions, axis=-1)
    output_array = output_array.write(i+1, predicted_id[0])
    if predicted_id == end:
        break

output = tf.transpose(output_array.stack())
