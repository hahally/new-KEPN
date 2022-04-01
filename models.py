import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import *

import numpy as np
import random

random.seed(2021)
np.random.seed(2021)
tf.random.set_seed(2021)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

# multi head attention
class MHA(layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, dropout_rate=0.3):
        super(MHA, self).__init__()
        
        # self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.multi = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=True):
        q,k,v,attention_mask = inputs
        # attn_output = self.att(q, v, k, attention_mask = attention_mask*-1e9)
        attn_output, attention_weights = self.multi(v, k, q, attention_mask)
        attn_output = self.dropout(attn_output, training=training)
        out = self.layernorm(q + attn_output)
        
        return out


class FFN(layers.Layer):
    def __init__(self, dropout_rate=0.3, embed_dim=512, ff_dim=2048):
        super(FFN, self).__init__()
        
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),layers.Dense(embed_dim),]
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training=True):
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, training=training)
        out = self.layernorm(x + ffn_output)

        return out

class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.3):
        super(EncoderLayer, self).__init__()
        
        self.multihead_attention = MHA(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn = FFN(dropout_rate=dropout_rate, embed_dim=embed_dim, ff_dim=ff_dim)
        
    def call(self, inputs, training=True):
        enc, attention_mask = inputs
        enc = self.multihead_attention((enc,enc,enc,attention_mask),training=training)
        enc = self.ffn(enc, training=training)
        
        return enc

class DecoderLayer(layers.Layer):
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.3):
        super(DecoderLayer, self).__init__()
        
        self.multihead_attention_1 = MHA(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
        self.multihead_attention_2 = MHA(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate)
        self.ffn = FFN(dropout_rate=dropout_rate, embed_dim=embed_dim, ff_dim=ff_dim)
        
    def call(self, inputs, training=True):
        dec, memory, look_ahead_mask, padding_mask = inputs
        dec = self.multihead_attention_1((dec,dec,dec,look_ahead_mask),training=training)
        dec = self.multihead_attention_2((dec,memory,memory,padding_mask),training=training)
        dec = self.ffn(dec, training=training)
        
        return dec
   

class Transformer(tf.keras.Model):
    def __init__(self, maxlen=50, vocab_size=32000, embed_dim=512, num_heads=8, ff_dim=2048, dropout_rate=0.3, num_blocks=6, paraphrase_type=1):
        super(Transformer, self).__init__()
        
        self.num_blocks = num_blocks
        self.d_model = embed_dim
        
        self.paraphrase_type = 1
        
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = positional_encoding(position= maxlen+2, d_model=embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        
        # g(y_, si) = V.T@tanh(W*[y_, si])
        self.si_attention_weight_w = layers.Dense(2*self.d_model, activation='tanh', kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01), use_bias=False)
        self.si_attention_weight_v = layers.Dense(2*self.d_model, kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01), use_bias=False)
        
        # g(y_, pi) = V.T@tanh(W*[y_, pi])
        self.pi_attention_weight_w = layers.Dense(2*self.d_model, activation='tanh', kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01), use_bias=False)
        self.pi_attention_weight_v = layers.Dense(2*self.d_model, kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01), use_bias=False)
        
        # y_t = softmax(Wy*[y_,ct])
        self.dense = layers.Dense(self.d_model, activation='tanh', kernel_initializer=tf.initializers.random_normal(mean=0, stddev=0.01), use_bias=False)
        # Final linear projection (embedding weights are shared)
        # 这里用一个dense层
        self.final_layer = layers.Dense(vocab_size, activation='softmax', kernel_initializer=tf.initializers.GlorotUniform())
        
        self.enc_layers = [EncoderLayer(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        self.dec_layers = [DecoderLayer(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        
        # synonym_label = [0,1,0,1,2,2,2,2], 其中 pad = 2,为填充值，做labeling任务时将其看成一个类别
        self.labeling_dense = layers.Dense(3, activation='softmax', kernel_initializer=tf.initializers.GlorotUniform())
        
    def call(self, inputs, training=True):
        encoder_inputs, decoder_inputs, x_paraphrased_dict = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(encoder_inputs, decoder_inputs)
        
        memory = self.encode(encoder_inputs, attention_mask=enc_padding_mask, training=training)
        synonym_labeling_out = self.labeling(memory)
        
        dec = self.decode(decoder_inputs, memory, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask,training=training)
        
        # add paraphrased dictionary attention
        inputs = (dec, decoder_inputs, x_paraphrased_dict)
        ct = self.get_attention_info(inputs)
        
        out = tf.concat([dec,ct], axis=-1)
        out = self.dense(out)
        out = self.final_layer(out)
        
        return out, synonym_labeling_out

    def encode(self, enc, attention_mask=None, training=True):
        seq_len = tf.shape(enc)[1]
        enc = self.token_emb(enc)
        enc *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        enc += self.pos_encoding[:,:seq_len,:]
        enc = self.dropout(enc,training=training)
        
        for i in range(self.num_blocks):
            inputs = (enc,attention_mask)
            enc = self.enc_layers[i](inputs,training=training)
            
        memory = enc
        
        return memory
    
    
    def decode(self, dec, memory, look_ahead_mask = None, padding_mask=None, training=True):
        seq_len = tf.shape(dec)[1]
        dec = self.token_emb(dec) 
        dec *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        dec += self.pos_encoding[:,:seq_len,:]
        dec = self.dropout(dec,training=training)
        
        for i in range(self.num_blocks):
            inputs = (dec, memory, look_ahead_mask, padding_mask)
            dec = self.dec_layers[i](inputs,training=training)
        
        return dec
    
    def labeling(self, memory):
        out = self.labeling_dense(memory)
        
        return out

    
    def get_paraphrased_emb(self, x_paraphrased_dict):
        x_paraphrased_o, x_paraphrased_p = x_paraphrased_dict[:,:,0], x_paraphrased_dict[:,:,1]
        x_paraphrased_o_embedding = self.token_emb(x_paraphrased_o)
        
        if self.paraphrase_type == 0:
            x_paraphrased_p_embedding = self.token_emb(x_paraphrased_p)
        if self.paraphrase_type == 1:
            x_paraphrased_p_embedding = tf.nn.embedding_lookup(self.pos_encoding[0], x_paraphrased_p)
        
        return x_paraphrased_o_embedding, x_paraphrased_p_embedding
    
    def get_attention_info(self, inputs):
        dec, decoder_inputs, x_paraphrased_dict = inputs
        batch_size = tf.shape(decoder_inputs)[0]
        seqlens = tf.shape(decoder_inputs)[1]
        paraphrased_lens = tf.shape(x_paraphrased_dict)[1]
        si_embedding, pi_embedding = self.get_paraphrased_emb(x_paraphrased_dict)
        
        # h = y*_t
        h = tf.fill([batch_size, seqlens, paraphrased_lens, self.d_model], 1.0) * tf.expand_dims(dec, axis=2)
        
        # si: a_(i,t)
        si_emb = tf.fill([batch_size, seqlens, paraphrased_lens, self.d_model], 1.0) * tf.expand_dims(si_embedding, axis=1)
        # 计算 g(y*_t, si)
        h_si_concat = tf.concat([h, si_emb], -1) # N, T2, W2, 2*d_model
        score_si = self.si_attention_weight_w(h_si_concat)
        score_si = self.si_attention_weight_v(score_si)
        score_si = tf.reduce_sum(score_si, axis=-1)
        
        # attention a: 得分
        a = tf.nn.softmax(score_si)
        
        # ct_1: ct的前半部分 a*si
        ct_si = tf.matmul(a, si_embedding) # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model
        
        # pi: a_(i,t)
        pi_emb = tf.fill([batch_size, seqlens, paraphrased_lens, self.d_model], 1.0) * tf.expand_dims(pi_embedding, axis=1)
        h_pi_concat = tf.concat([h,pi_emb], -1) # N, T2, W2, 2*d_model
        score_pi = self.pi_attention_weight_w(h_pi_concat)
        score_pi = self.pi_attention_weight_v(score_pi)
        score_pi = tf.reduce_sum(score_pi, axis=-1)
        a = tf.nn.softmax(score_pi)
        ct_pi = tf.matmul(a,pi_embedding) # (N, T2, W2) * (N, W2, d_model) --> N, T2, d_model
        
        ct = tf.concat([ct_si,ct_pi], axis=-1) # N, T2, d_model --> N, T2, 2*d_model
        
        return ct
        
    def create_masks(self, src, tar):
        enc_padding_mask = create_padding_mask(src)
        dec_padding_mask = create_padding_mask(src)
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, look_ahead_mask, dec_padding_mask
        
        
