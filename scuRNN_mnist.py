# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:38:34 2018

A Recurrent Neural Network (RNN) implementation using TensorFlow library.  Can
be used to run the complex scoRNN architecture and an LSTM.  Uses the MNIST database of 
handwritten digits (http://yann.lecun.com/exdb/mnist/).
"""


# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import sys
import time

from scuRNN import *


'''
To classify images using a recurrent neural network, we consider every image
as a sequence of single pixels. Because MNIST image shape is 28*28px, we will 
then handle 784 steps of single pixels for every sample.
'''
# Network parameters
model = 'scuRNN'
n_input = 1             # MNIST data input (single pixel at a time)
n_steps = 784           # Number of timesteps (img shape: 28*28)
n_hidden = 116          # Hidden layer size
n_classes = 10          # MNIST total classes (0-9 digits)
modreluflag = True      # Used for modrelu activation
permuteflag = False     # Used for permuted MNIST
training_epochs = 70
batch_size = 50
h_omax = 0.01


# Input/Output parameters
in_out_optimizer = 'adam'
in_out_lr = 1e-3


# Hidden to Hidden Parameters
A_optimizer = 'rmsprop'
A_lr = 1e-4

# theta training Parameters
Theta_optimizer = 'adagrad'
Theta_lr = 1e-3

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

bflagt = True
bflagf = False


# COMMAND LINE ARGS: MODEL HIDDENSIZE IO-OPT IO-LR AOPT ALR NEG-ONES
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    Theta_optimizer = sys.argv[7]
    Theta_lr = float(sys.argv[8])
    
except IndexError:
    pass


# Setting the random seed
tf.set_random_seed(5544)#1234
np.random.seed(5544)#1234    5544

# Assigning permutation, if applicable
if permuteflag:
    permute = np.random.RandomState(92916)
    xpermutation = permute.permutation(784)    
    x = tf.gather(x, xpermutation, axis=1)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

whole_test_data = mnist.test.images.reshape((-1, n_steps, n_input))
im_test = np.zeros((whole_test_data.shape[0], n_steps, n_input))
whole_test_label = mnist.test.labels

test_data = np.split(whole_test_data, 200)
test_label = np.split(whole_test_label, 200)

# Name of save string/scaling matrix
if model == 'LSTM':
    savestring = '{:s}_{:d}_{:s}_{:.1e}'.format(model, n_hidden, \
                 in_out_optimizer, in_out_lr)
if model == 'scuRNN':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_{:s}_{:.1e}'.format(model, \
                 n_hidden, in_out_optimizer, in_out_lr, \
                 A_optimizer,A_lr,Theta_optimizer,Theta_lr)     #     
    
print('\n')
print(savestring)
print('\n')

# Defining RNN architecture
def RNN(x):

    # Create RNN cell
    if model == 'LSTM':
        rnn_cell = rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
        # Place RNN cell into RNN, take last timestep as output
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
        rnnoutput = outputs[:,-1]
    
    if model == 'scuRNN':
        #D = make_D(theta)
        rnn_cell = scuRNNCell(n_input,n_hidden)
        #tf.random_uniform([5], 0, 10, dtype=tf.int32, seed=0))
        
        state_init = tf.get_variable("ho", [1,2*n_hidden], \
                     initializer=init_ops.random_uniform_initializer(-0.01, 0.01))

        #state_init = tf.Variable(np.random.uniform(-h_omax,h_omax,size = [1, 2*n_hidden]).astype(np.float32),trainable=True)
        Initial_state = tf.tile(state_init, [batch_size, 1])
        
        #state_init = np.random.uniform(-0.01,0.01,size = [1, 2*n_hidden]).astype(np.float32)
        #Initial_state = tf.tile(state_init, [batch_size, 1])
        
        # Place RNN cell into RNN, take last timestep as output
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=Initial_state, dtype=tf.float32)   #
        rnnoutput = outputs[:,-1]
    
    # Last layer, linear
    output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None,\
                             kernel_initializer=init_ops.glorot_uniform_initializer(seed=5544,dtype=tf.float32))
    
    
    return output,states,Initial_state

#Initialize theta
def theta_init(n_hidden):
    theta = np.random.uniform(0,(2.0)*np.pi,size = n_hidden)
        
    return theta

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

# Plotting loss & accuracy
def graphlosses(xax, tr_loss, te_loss, tr_acc, te_acc):
       
    plt.subplot(2,1,1)
    plt.plot(xax, tr_loss, label='training loss')
    plt.plot(xax, te_loss, label='testing loss')
    plt.ylim([0,3])
    plt.legend(loc='lower left', prop={'size':6})
    plt.subplot(2,1,2)
    plt.plot(xax, tr_acc, label='training acc')
    plt.plot(xax, te_acc, label='testing acc')
    plt.ylim([0,1])
    plt.legend(loc='lower left', prop={'size':6})
    plt.savefig(savestring + '.png')
    plt.clf()
    
    return

# Used for printing values
def getprintvalues(A, W):
    I = np.identity(A.shape[0])
    orthogonalcheck = np.linalg.norm(I - np.dot(np.conjugate(W).T,W))  
    A_norm = np.linalg.norm(A, ord=1)
    IA_inverse = np.linalg.lstsq(I + A, I)
    IA_inverse_norm = np.linalg.norm(IA_inverse[0], ord=1)
    IW_norm = np.linalg.norm(I + W, ord=1)
    
    return orthogonalcheck, A_norm, IA_inverse_norm, IW_norm

    # Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


# Assigning to RNN function
pred,last_state,IState= RNN(x)

    
# Define loss object
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, \
       labels=y))


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Optimizers/Gradients
optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                  'adagrad' : tf.train.AdagradOptimizer,
                  'rmsprop' : tf.train.RMSPropOptimizer,
                  'sgd' : tf.train.GradientDescentOptimizer}

opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)
opt2 = optimizer_dict[Theta_optimizer](learning_rate=Theta_lr)

# LSTM training operations
if model == 'LSTM':
    LSTMtrain = opt1.minimize(loss)

# scoRNN training operations
if model == 'scuRNN':
    Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
    bvar = [v for v in tf.trainable_variables() if 'b:0' in v.name][0]
    Uvar = [v for v in tf.trainable_variables() if 'U:0' in v.name][0]
    Thetavar = [v for v in tf.trainable_variables() if 'T:0' in v.name][0]
    Vvar = [v for v in tf.trainable_variables() if 'dense/kernel:0' in v.name][0]
    cvar = [v for v in tf.trainable_variables() if 'dense/bias:0' in v.name][0]
    hovar = [v for v in tf.trainable_variables() if 'ho:0' in v.name][0]
    
    scuRNNtrain = opt1.minimize(loss,var_list = [bvar,Uvar,Vvar,cvar,hovar])
    
    grads_W = tf.gradients(loss,[Wvar])[0]
    
    # Updating variables
    newW = tf.placeholder(tf.float32, Wvar.get_shape())
    grad_Theta = tf.placeholder(tf.float32,Thetavar.get_shape())
    applygradtheta = opt2.apply_gradients([(grad_Theta, Thetavar)])
    updateW = tf.assign(Wvar, newW)
    
# Plotting lists
epochs_plt = []
train_loss_plt = []
test_loss_plt = []
train_accuracy_plt = []
test_accuracy_plt = []


# Training
with tf.Session() as sess:
    
    # Initializing the variables
    t = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    A = A_init(n_hidden)
    theta = sess.run(Thetavar)
    I = np.identity(A.shape[0])
    IplusAinv = np.linalg.lstsq(I+A,I)[0]   
    IminusA = I-A
    Dr,D = make_D(theta)
    W = makeW(A, IplusAinv,D)
    W_expanded = Expand_Matrix(W)
    sess.run(updateW, feed_dict = {newW: W_expanded})

    # Keep training until reach number of epochs
    epoch = 1
    while epoch <= training_epochs:                          
        step = 1
        # Keep training until reach max iterations
        #print("max iterations :",mnist.train.images.shape[0])
        while step * batch_size <= mnist.train.images.shape[0]:

            # Getting input data
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            # Reshape data to get 784 seq of 1 pixel
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            
                             
            # Updating weights
            if model == 'LSTM':
                sess.run(LSTMtrain, feed_dict={x: batch_x, y: batch_y})         
            if model == 'scuRNN':          
                values,IS= sess.run([last_state,IState], feed_dict = {x: batch_x})
                                
                _,hidden_grads,L = sess.run([scuRNNtrain, grads_W,loss], \
                                 feed_dict = {x: batch_x, y: batch_y})
                
                reduced_hidden_grads = Reduce_Matrix(hidden_grads)
                #print("reduce hidden gradients",reduced_hidden_grads)
                DFA,DFT = Cayley_Transform_Deriv(reduced_hidden_grads, IplusAinv,IminusA, W,Dr,D) 
                #print("DFT shape :",DFT.shape, "DFT :",DFT)
                sess.run(applygradtheta, feed_dict = {grad_Theta: DFT})
                theta = sess.run(Thetavar)
                   
                # Update A matrix by standard gradient descent  # Use other optimizers
                A,tA,sA,rA,sAi,rAi = Update_step_A(A_lr,A,DFA,A_optimizer,tA,sA,rA,sAi,rAi)
                IminusA = I-A
                Dr,D = make_D(theta)
                IplusAinv = np.linalg.lstsq(I+A,I)[0]  
                W = makeW(A, IplusAinv,D)
                W_expanded = Expand_Matrix(W)
                sess.run(updateW, feed_dict = {newW: W_expanded})
                
            step += 1
            

        # Evaluating average epoch accuracy/loss of model
        test_acc, test_loss = map(list, zip(*[sess.run([accuracy, loss], \
                   feed_dict={x: tbatch, y: tlabel}) \
                   for tbatch, tlabel in zip(test_data, test_label)]))
        test_acc, test_loss = np.mean(test_acc), np.mean(test_loss)

        # Evaluating training accuracy/loss of model on random training batch               
        train_index = np.random.randint(0, 1051)
        train_x = mnist.train.images[train_index:train_index + batch_size,:]
        train_y = mnist.train.labels[train_index:train_index + batch_size,:]
        train_x = train_x.reshape((batch_size, n_steps, n_input))
        
        train_acc, train_loss = sess.run([accuracy, loss], \
                                feed_dict={x: train_x, y: train_y})

        
        # Printing results
        print('\n')
        print("Completed Epoch: ", epoch)
        print("Testing Accuracy:", test_acc)
        print("Testing Loss:", test_loss)
        print("Training Accuracy:", train_acc)
        print("Training Loss:", train_loss)
        print('\n')
        
        # Plotting results
        epochs_plt.append(epoch)
        train_loss_plt.append(train_loss)
        test_loss_plt.append(test_loss)
        train_accuracy_plt.append(train_acc)
        test_accuracy_plt.append(test_acc)
               
        graphlosses(epochs_plt, train_loss_plt, test_loss_plt, \
                    train_accuracy_plt, test_accuracy_plt)
        
        # Saving files
        np.savetxt(savestring + '_train_loss.csv', \
                   train_loss_plt, delimiter = ',')
        np.savetxt(savestring + '_test_loss.csv', \
                   test_loss_plt, delimiter = ',')
        np.savetxt(savestring + '_train_acc.csv', \
                   train_accuracy_plt, delimiter = ',')
        np.savetxt(savestring + '_test_acc.csv', \
                test_accuracy_plt, delimiter = ',')
        
                   
        epoch += 1
    
    print("Optimization Finished!" ,time.time()-t)        

