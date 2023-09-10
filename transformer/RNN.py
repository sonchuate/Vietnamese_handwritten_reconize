from tensorflow.keras import Sequential, Model, layers
from tensorflow.keras.layers import Add, Conv1D, MaxPooling2D,\
      BatchNormalization, MaxPool2D, Input, Dense, Softmax,\
      Rescaling, Dropout, LSTM, Bidirectional, Embedding
import tensorflow as tf

class GLU(tf.keras.layers.Layer):
    #GLU layer
    def __init__(self):
        super(GLU, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        dim = input_shape[-1] // 2
        a, b = inputs[:, :, :dim], inputs[:, :, dim:]

        b = tf.sigmoid(b)
        return a * b

def encoder(src):
    max_length = 32
    drop_pob = 0.1
    emb_dim =  src.shape[1]
    hid_dim = 128
    num_conv_block = 4
    kernel_size = 3
    global GLU
    GLU = GLU()
    scale = tf.constant([0.707])
    batch_size = 1

    pos = tf.range(0, emb_dim)
    pos = tf.tile(pos, [1])  #[batch_size, src len]
    pos = Embedding(max_length, emb_dim)(pos) #[batch_size, src len, emb_dim]
    embedded = Dropout(drop_pob)(pos + src) #[batch_size, src len, emb_dim]
    conv_input = Dense(hid_dim)(embedded) #[batch_size, src len, hid_dim]
    # conv_input = tf.transpose(conv_input, perm=[1, 0])
    # for i in range(num_conv_block):
    #     conved = Conv1D(filters = hid_dim * 2, kernel_size = kernel_size)(conv_input)
    #     conved = GLU(conved)
    #     conved = (conved + conv_input) * scale
    #     conv_input = conved
    # conved = Dense(emb_dim)(conv_input)
    # combined = (conved + embedded) * scale
    # return conved, combined
    return conv_input
class RecBlock(tf.keras.layers.Layer):
    def __init__(self, vocab_len, prob_drop = 0.1,**kwargs):

        super().__init__()
        self.bidirectional = Bidirectional(LSTM(96, return_sequences=True))
        self.dropout = Dropout(prob_drop)
        self.dense = Dense(vocab_len + 1, activation="softmax", name="output")
        def transpose1(a):
            return tf.transpose(a, perm=[0, 2, 1])
        self.transpose1 = transpose1


    def call(self, inputs):
        #b c t
        rec = self.bidirectional(inputs)
        #b c t'
        rec = self.dropout(rec)
        #b c t'
        rec = self.transpose1(rec)
        #b t' c
        rec = self.dense(rec)
        #b t' (vocab+1)
        return rec



    

