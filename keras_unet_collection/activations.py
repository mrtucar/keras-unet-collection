
from tensorflow import math
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def gelu_(X):

    return 0.5*X*(1.0 + math.tanh(0.7978845608028654*(X + 0.044715*math.pow(X, 3))))

def snake_(X, beta):

    return X + (1/beta)*math.square(math.sin(beta*X))

def q_relu(x):
  if x>0:
    x = x
    return x
  else:
    x = 0.01*x-2*x
    return x

# Vectorising the QReLU function
np_q_relu = np.vectorize(q_relu)

# Defining the derivative of the function QReLU
def d_q_relu(x):
  if x>0:
    x = 1
    return x
  else:
    x = 0.01-2
    return x

# Vectorising the derivative of the QReLU function
np_d_q_relu = np.vectorize(d_q_relu)

# Defining the gradient function of the QReLU
def q_relu_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_q_relu(x)
    return grad * n_gr

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
# Generating a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

np_q_relu_32 = lambda x: np_q_relu(x).astype(np.float32)

def tf_q_relu(x,name=None):
    with tf.name_scope(name, "q_relu", [x]) as name:
        y = py_func(np_q_relu_32,  # Forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= q_relu_grad)  # The function that overrides gradient
        y[0].set_shape(x.get_shape())  # To specify the rank of the input.
        return y[0]

np_d_q_relu_32 = lambda x: np_d_q_relu(x).astype(np.float32)

def tf_d_q_relu(x,name=None):
    with tf.name_scope(name, "d_q_relu", [x]) as name:
        y = tf.py_func(np_d_q_relu_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]


class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a tf.keras.Layer
    
    
    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

    
class Snake(Layer):
    '''
    Snake activation function $X + (1/b)*sin^2(b*X)$. Proposed to learn periodic targets.
    
    Y = Snake(beta=0.5, trainable=False)(X)
    
    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions 
    and how to fix it. arXiv preprint arXiv:2006.08195.
    
    '''
    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class QReLU(Layer):

    def __init__(self,**kwargs):
        super(QReLU,self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,name=None):
        return tf_q_relu(inputs,name=None)

    def get_config(self):
        base_config = super(QReLU, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape