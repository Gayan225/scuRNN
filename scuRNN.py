# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:34:32 2018
scoRNNCell 
@author: gayan
"""

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops


class scuRNNCell(RNNCell):
    """Scaled Cayley Orthogonal Recurrent Network (scoRNN) Cell

    """
    def __init__(self, input_size,hidden_size, activation='modReLU'): 
        
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation

        # Initialization of hidden to hidden matrix       
        self._W = tf.get_variable("W", [self._hidden_size, 2*self._hidden_size], \
                  initializer = init_ops.constant_initializer())
        
        # Initialization of hidden bias
        self._bias = tf.get_variable("b", [self._hidden_size], \
                     initializer=init_ops.random_uniform_initializer(-0.01, 0.01)) 

        # Initialization theta to create diagonal matrix
        self._theta = tf.get_variable("T", [self._hidden_size], \
                     initializer=init_ops.random_uniform_initializer(0, (2.0)*np.pi)) 
    
    @property
    def state_size(self):
        return 2*self._hidden_size  # Double the hidden size for complex 

    @property
    def output_size(self):
        return 2*self._hidden_size  # Double the hidden size for complex

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "scuRNNCell"):
            
                      
            # Initialization of input matrix
            U_init = init_ops.glorot_uniform_initializer(seed=5544,dtype=tf.float32)
            U = vs.get_variable("U", [self._input_size, 2*self._hidden_size], initializer= U_init)
 

            # Expanding input/hidden matrix for complex multiplications
            Real_U = U[:,0:self._hidden_size]
            Imag_U = U[:,self._hidden_size:]
            Un = tf.concat((Real_U,Imag_U), axis=1)
            
            # Expanding hidden/hidden matrix for complex multiplications                      
            Real_W = self._W[0:self._hidden_size,0:self._hidden_size]
            Imag_W = self._W[0:self._hidden_size, self._hidden_size:]
            W_Row = tf.concat((-Imag_W, Real_W), axis=1)
            Expanded_W = tf.concat((self._W, W_Row), axis=0)
            
            # Forward pass of graph          
            lin_output = math_ops.matmul(inputs, Un) + math_ops.matmul(state, Expanded_W)
                        
            # Applying modReLU
            if self._activation == 'modReLU':           
                
                Real_res = lin_output[:,0:self._hidden_size]
                Complex_res = lin_output[:, self._hidden_size:]
                #z = tf.complex(Real_res,Complex_res)
                modulus = tf.sqrt(Real_res*Real_res + Complex_res*Complex_res+1e-5)
                scale = tf.nn.relu(nn_ops.bias_add(modulus, self._bias))/(modulus+1e-5) 
                extended_scale = tf.concat((scale,scale),axis =1)
                output = lin_output*extended_scale
                
            else:
                output = self._activation(nn_ops.bias_add(lin_output, self._bias))
                
        return output, output
        
