'''
Created on Sun Jun 10 10:23:34 2018
A Recurrent Neural Network (RNN) implementation example using TensorFlow library.
Input is a sequence of digits to copy, followed by a string of T 0s, 
a marker (we use 9) to begin printing the copied sequence, and more zeroes.
Target output is all zeroes until the marker, at which point the machine
should print the sequence it saw at the beginning.

Example for T = 10 and n_sequence = 5:

Input
3 6 5 7 2 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0
Target output
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 6 5 7 2
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pylab as plt
import scipy.sparse as sps

import sys
import os
import time

from scuRNN import *





# Network Parameters
model = 'scuRNN'
T = 2000          # number of zeroes to put between sequence and marker
n_sequence = 10  # length of sequence to copy
n_input = 10          # number of possible inputs (0-9)
n_classes = 9          # number of possible output classes (0-8, 9 is not a valid output)
n_hidden = 130          # hidden layer num of features
train_size = 20000
test_size = 1000
batch_size = 20
display_step = 50
iterations = 2000

# Input/Output Parameters
in_out_optimizer = 'adam'
in_out_lr = 1e-3

# Hidden to Hidden Parameters
A_optimizer = 'rmsprop'
A_lr = 1e-4

# theta training Parameters
Theta_optimizer = 'adagrad'
Theta_lr = 1e-4


# COMMAND LINE ARGS: MODEL TFOPT AOPT TFLR ALR HIDDENSIZE NEGEIGS MRELU
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    Theta_optimizer = sys.argv[7]
    Theta_lr = float(sys.argv[8])
    T = int(sys.argv[9])
except IndexError:
    pass

# Custom Optimizer Parameters
rho1 = 0.9
rho2 = 0.999
delta = 1e-8
#Use to Update A  
tA=0
sA = np.zeros((n_hidden, n_hidden))
rA = np.zeros((n_hidden, n_hidden))
sAi = np.zeros((n_hidden, n_hidden))
rAi = np.zeros((n_hidden, n_hidden))

n_steps = n_sequence*2 + T

baseline = (n_sequence*np.log(n_classes-1))/(T + 2*n_sequence)

A = np.zeros((n_hidden, n_hidden))

# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Plotting Commands
number_iters_plt = []
train_loss_plt = []
test_loss_plt = []
train_accuracy_plt = []
test_accuracy_plt = []


# name of graph file
if model == 'LSTM':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_T={:d}'.format(model, n_hidden, \
                 in_out_optimizer, in_out_lr, T)
if model == 'scuRNN':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_{:s}_{:.1e}_T={:d}'.format(model, \
                 n_hidden, in_out_optimizer, in_out_lr, \
                 A_optimizer, A_lr, Theta_optimizer,Theta_lr,  T)
    
print('\n' + savestring + '\n')


def copying_data(T, seq):
    n_data, n_sequence = seq.shape
    
    zeros1 = np.zeros((n_data, T-1))
    zeros2 = np.zeros((n_data, T))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))
    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
    
    return x, y


# Defining RNN architecture
def RNN(x):

    # Create RNN cell
    if model == 'LSTM':
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
        # Place RNN cell into RNN, take last timestep as output
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    
    if model == 'scuRNN':
        rnn_cell = scuRNNCell(n_input,n_hidden)
        state_init = tf.get_variable("ho", [1,2*n_hidden], \
                     initializer=init_ops.random_uniform_initializer(-0.01, 0.01))

        #state_init = tf.Variable(np.random.uniform(-h_omax,h_omax,size = [1, 2*n_hidden]).astype(np.float32),trainable=True)
        Initial_state = tf.tile(state_init, [batch_size, 1])
    
        # Place RNN cell into RNN, take last timestep as output
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x,initial_state=Initial_state,  dtype=tf.float32)   #
    #rnnoutput = outputs[:,-1]
    
    with tf.variable_scope("output"):
        weights = tf.get_variable("weights", shape=[2*n_hidden, n_classes])
        biases = tf.get_variable("bias", shape=[n_classes])
    
    temp_out = tf.map_fn(lambda r: tf.matmul(r, weights), outputs)
    output_data = tf.nn.bias_add(temp_out, biases)
    return output_data
    
    
 
#######################################
def make_D(theta):
    Dr = np.cos(theta) +1j*np.sin(theta)
    D  = np.diag(Dr)

    return Dr,D  


# Initialization of skew-Hermitian matrix A
def A_init(n_hidden):
    
    # Create real-symmetric component of A
    s = np.random.uniform(0, np.pi/2.0, \
        size=int(np.floor(n_hidden/2.0)))
    s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))
    z = np.zeros(s.size)
    if n_hidden % 2 == 0:
        diag = np.hstack(zip(s, z))[:-1]
    else:
        diag = np.hstack(zip(s,z))
    Real_A = np.diag(diag, k=1)
    Real_A = Real_A - Real_A.T
    
    #Real_A = np.diag(np.zeros(n_hidden))

    imag_A = np.diag(1J*np.zeros(n_hidden))
    
    #create A
    A = Real_A + imag_A
    
    return A


# Used to make the hidden to hidden weight matrix
def makeW(A, IplusAinv,D):
    # Computing hidden to hidden matrix using the relation 
    # W = (I + A)^-1*(I - A)D
    
    I = np.identity(A.shape[0])
    W = np.dot(np.dot(IplusAinv, I - A),D)
            
    return W   


# Expanding matrix into real and imaginary parts
def Expand_Matrix(matrix):
    expanded = np.concatenate((matrix.real, matrix.imag), axis=1)
    
    return expanded


# Reducing matrix into complex matrix (when matric contans the real part of 
#the gredient and imaginary part of the matrix)
def Reduce_Matrix(matrix):
    Real_matrix = matrix[0:n_hidden, 0:n_hidden]
    Imag_matrix = matrix[0:n_hidden, n_hidden:]
    
    Combined_matrix = (1/2.0)*Real_matrix - 1j*(1/2.0)*Imag_matrix
    
    return Combined_matrix     


# Used to calculate Cayley Transform derivative
def Cayley_Transform_Deriv(grads, IplusAinv,IminusA, W,Dr,D):
    
    # Calculate update matrix
    Update = np.dot(np.dot(IplusAinv.T, grads), D + W.T)
    DFA = (Update).T - np.conjugate(Update)
    
    Z = np.dot(IplusAinv,IminusA)
    I1 = np.identity(IplusAinv.shape[0])
    updateTeata = 1j*np.dot((np.multiply(np.dot(grads.T,Z),I1)),Dr.T)
    DFtheata = (2.0)*np.real(updateTeata)
       
    return DFA,DFtheata
    
def Update_step_A(step_size,A,gradients,updatetype,t,s,r,si,ri):
     gradreal = np.real(gradients)
     gradimag = np.imag(gradients)
     if updatetype == 'adam':
         t = t+1
         s = rho1*s + (1-rho1)*gradreal
         r = rho2*r + (1-rho2)*gradreal*gradreal
         shat = (1.0/(1-rho1**t))*s
         rhat = (1.0/(1 - rho2**t))*r
         Update_R = -step_size*np.reciprocal(np.sqrt(rhat)+delta)*shat
         si = rho1*si + (1-rho1)*gradimag
         ri = rho2*ri + (1-rho2)*gradimag*gradimag
         shati = (1.0/(1-rho1**t))*si
         rhati = (1.0/(1 - rho2**t))*ri
         Update_I = -step_size*np.reciprocal(np.sqrt(rhati)+delta)*shati
         Update = Update_R +1j*Update_I
        

     if updatetype == 'rmsprop':
         r = rho1*r + (1-rho1)*gradreal*gradreal
         Update_R = -step_size*np.reciprocal(np.sqrt(r+delta))*gradreal
         ri = rho1*ri + (1-rho1)*gradimag*gradimag
         Update_I = -step_size*np.reciprocal(np.sqrt(ri+delta))*gradimag
         Update = Update_R +1j*Update_I

     if updatetype == 'adagrad':
         r += gradreal*gradreal
         Update_R = -step_size*np.reciprocal(np.sqrt(r) + delta)*gradreal
         ri += gradimag*gradimag
         Update_I = -step_size*np.reciprocal(np.sqrt(ri) + delta)*gradimag
         Update = Update_R +1j*Update_I

     if updatetype == 'sgd':
         Update_R = -step_size*gradreal
         Update_I = -step_size*gradimag
         Update = Update_R +1j*Update_I
        
       
     newParam =A + Update   
     
     return newParam,t,s,r,si,ri


# Used for printing values
def getprintvalues(A, W):
    I = np.identity(A.shape[0])
    orthogonalcheck = np.linalg.norm(I - np.dot(W.T,W))
    A_norm = np.linalg.norm(A, ord=1)
    IA_inverse = np.linalg.lstsq(I + A, I)
    IA_inverse_norm = np.linalg.norm(IA_inverse[0], ord=1)
    IW_norm = np.linalg.norm(I + W, ord=1)
    
    return orthogonalcheck, A_norm, IA_inverse_norm, IW_norm


# Plotting loss & accuracy
def graphlosses(xax, tr_loss, te_loss, tr_acc, te_acc):
       
    plt.subplot(2,1,1)
    plt.plot(xax, tr_loss, label='training loss')
    plt.plot(xax, te_loss, label='testing loss')
    plt.ylim([0,baseline*2])
    plt.legend(loc='lower left', prop={'size':6})
    plt.subplot(2,1,2)
    plt.plot(xax, tr_acc, label='training acc')
    plt.plot(xax, te_acc, label='testing acc')
    plt.ylim([0.8,1])
    plt.legend(loc='lower left', prop={'size':6})
    plt.savefig(savestring + '.png')
    plt.clf()
    
    return

# Graph input
x = tf.placeholder("int32", [None, n_steps])
y = tf.placeholder("int64", [None, n_steps])
    
inputdata = tf.one_hot(x, n_input, dtype=tf.float32)

#inputdata = tf.one_hot(x, n_input, dtype=tf.float32)

# Assigning variable to RNN function
pred = RNN(inputdata)

# Cost & accuracy
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))



correct_pred = tf.equal(tf.argmax(pred, 2), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Define optimizer object
optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                'adagrad' : tf.train.AdagradOptimizer,
                'rmsprop' : tf.train.RMSPropOptimizer,
                'sgd' : tf.train.GradientDescentOptimizer}

opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)
opt2 = optimizer_dict[Theta_optimizer](learning_rate=Theta_lr)

if model == 'LSTM':
    LSTMtrain = opt1.minimize(cost)

if model == 'scuRNN':
    Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
    bvar = [v for v in tf.trainable_variables() if 'b:0' in v.name][0]
    Uvar = [v for v in tf.trainable_variables() if 'U:0' in v.name][0]
    Thetavar = [v for v in tf.trainable_variables() if 'T:0' in v.name][0]
    hovar = [v for v in tf.trainable_variables() if 'ho:0' in v.name][0]
    othervarlist = [v for v in tf.trainable_variables() if v not in [Wvar,bvar,Uvar,Thetavar,hovar]][0]
    scuRNNtrain = opt1.minimize(cost,var_list = [bvar,Uvar,hovar, othervarlist])  
    grads_W = tf.gradients(cost,[Wvar])[0]
    
    # Updating variables
    newW = tf.placeholder(tf.float32, Wvar.get_shape())
    grad_Theta = tf.placeholder(tf.float32,Thetavar.get_shape())
    applygradtheta = opt2.apply_gradients([(grad_Theta, Thetavar)])
    updateW = tf.assign(Wvar, newW)
        
with tf.Session() as sess:
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Get initial A and W
    if model == 'scuRNN':
        A = A_init(n_hidden)
        theta = sess.run(Thetavar)
        I = np.identity(A.shape[0])
        IplusAinv = np.linalg.lstsq(I+A,I)[0]  
        IminusA = I-A
        Dr,D = make_D(theta)
        W = makeW(A, IplusAinv,D)
        W_expanded = Expand_Matrix(W)
        sess.run(updateW, feed_dict = {newW: W_expanded})
    
        
    training_seq = np.random.randint(1, high=9, size=(train_size, n_sequence))
    test_seq = np.random.randint(1, high=9, size=(test_size, n_sequence))
    test_seq = np.split(test_seq, 50)
        
    # Keep training until reach number of iterations
    step = 0
    while step  < iterations:
        # input data
        batch_seq = training_seq[(step*batch_size) % train_size:(step*batch_size)% train_size + batch_size]
        batch_x, batch_y = copying_data(T, batch_seq)
               
        # Updating weights
        if model == 'LSTM':
            sess.run(LSTMtrain, feed_dict={x: batch_x, y: batch_y})


        if model == 'scuRNN':
            _,hidden_grads,L = sess.run([scuRNNtrain, grads_W,cost], \
                                 feed_dict = {x: batch_x, y: batch_y})
            reduced_hidden_grads = Reduce_Matrix(hidden_grads)
            DFA,DFT = Cayley_Transform_Deriv(reduced_hidden_grads, IplusAinv,IminusA, W,Dr,D) 
            sess.run(applygradtheta, feed_dict = {grad_Theta: DFT})
            theta = sess.run(Thetavar)
            A,tA,sA,rA,sAi,rAi = Update_step_A(A_lr,A,DFA,A_optimizer,tA,sA,rA,sAi,rAi )
            IminusA = I-A
            Dr,D = make_D(theta)
            IplusAinv = np.linalg.lstsq(I+A,I)[0]  
            W = makeW(A, IplusAinv,D)
            W_expanded = Expand_Matrix(W)
            sess.run(updateW, feed_dict = {newW: W_expanded})
            
        step += 1
        
        if step % display_step == 0:
                    
            # Printing commands
            #print("hi")
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            
            test_metrics = []
            for seq in test_seq:
                tseq_x, tseq_y = copying_data(T, seq)
                test_metrics.append(sess.run([accuracy, cost], feed_dict={x: tseq_x, y: tseq_y}))
            test_acc, test_loss = np.mean(test_metrics, axis=0)
            #print("hi 2")
            print('\n')
            print("Iterations: ", step)
            print("Testing Accuracy:", test_acc)
            print("Testing Loss:", test_loss)
            print("Training Accuracy:", acc)
            print("Training Loss:", loss)
            print("Baseline:", baseline)
            print('\n')
            
            # Plotting
            number_iters_plt.append(step)
            train_loss_plt.append(loss)
            test_loss_plt.append(test_loss)
            train_accuracy_plt.append(acc)
            test_accuracy_plt.append(test_acc)
            
            graphlosses(number_iters_plt,train_loss_plt, test_loss_plt, train_accuracy_plt, test_accuracy_plt)
            
            # Saving files
            np.savetxt(savestring + '_train_loss.csv', \
                       train_loss_plt, delimiter = ',')
            np.savetxt(savestring + '_test_loss.csv', \
                       test_loss_plt, delimiter = ',')
            np.savetxt(savestring + '_train_acc.csv', \
                       train_accuracy_plt, delimiter = ',')
            np.savetxt(savestring + '_test_acc.csv', \
                       test_accuracy_plt, delimiter = ',')
            
print("Optimization Finished!")
        
