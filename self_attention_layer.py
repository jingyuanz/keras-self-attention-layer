from keras.layers import Layer
from keras import backend as K

class selfAttention(Layer):
    def __init__(self, n_head, hidden_dim, penalty=0.1, **kwargs):
        self.n_head = n_head
        self.P = penalty
        
        self.hidden_dim = hidden_dim
        super(selfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W1 = self.add_weight(name='w1', shape=(input_shape[2], self.hidden_dim), initializer='uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(self.hidden_dim, self.n_head), initializer='uniform',
                                  trainable=True)
        super(selfAttention, self).build(input_shape)
    
    def call(self, x, **kwargs):
        d1 = K.dot(x, self.W1)
        tanh1 = K.tanh(d1)
        d2 = K.dot(tanh1, self.W2)
        softmax1 = K.softmax(d2, axis=0)
        A = K.permute_dimensions(softmax1, (0, 2, 1))
        emb_mat = K.batch_dot(A, x, axes=[2, 1])
        reshape = K.batch_flatten(emb_mat)
        eye = K.eye(self.n_head)
        prod = K.batch_dot(softmax1, A, axes=[1, 2])
        self.add_loss(self.P * K.sqrt(K.sum(K.square(prod - eye))))
        return reshape
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] * self.n_head,)