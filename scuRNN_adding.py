# -*- coding: utf-8 -*-
'''
A Recurrent Neural Network (RNN) implementation using TensorFlow library.
Implements the adding problem.  Two sequences are fed into the RNN.  One
sequence is a random sequence of numbers and the other is a marker of zeros and 
ones.  The digits marked with one are to be added together.
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pylab as plt


import sys
import os
import time

from scuRNN import *

# Network Parameters
model = 'ScuRNN'
n_input = 2             
n_steps = 750           # length of sequence
n_hidden = 116          # hidden layer num of features
n_classes = 1           # one output (sum of two numbers)
modreluflag = True      # Used for modrelu activations
training_epochs = 30
batch_size = 50
display_step = 100
training_size = 100000   # Training set size
testing_size = 50    # Testing set size10000

# Input/Output Parameters
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

# COMMAND LINE ARGS: MODEL TFOPT TFLR AOPT ALR HIDDENSIZE NEGEIGS SQLENGTH
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    Theta_optimizer = sys.argv[7]
    Theta_lr = float(sys.argv[8])
    n_steps = int(sys.argv[9])
except IndexError:
    pass


# Setting the random seed
tf.set_random_seed(5544)
np.random.seed(5544)


# Plotting Commands
number_iters_plt = []
train_mse_plt = []
test_mse_plt = []
orthogonal_plt = []


# name of graph file
savestring = '{:s}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_{:s}_{:.1e}_sequence_length_{:d}.png'.format(model, n_hidden , in_out_optimizer, in_out_lr, A_optimizer, A_lr,Theta_optimizer,Theta_lr, n_steps)
graphstring = savestring + '.png'
print('\n')
print(graphstring)
print('\n')


# Generates Synthetic Data
def Generate_Data(size, length):
    
    # Random sequence of numbers
    x_random = np.random.uniform(0,1, size = [size, length])

    # Random sequence of zeros and ones
    x_placeholders = np.zeros((size, length))
    firsthalf = int(np.floor((length-1)/2.0))
    for i in range(0,size):
        x_placeholders[i, np.random.randint(0, firsthalf)] = 1
        x_placeholders[i, np.random.randint(firsthalf, length)] = 1

    # Create labels
    y_labels = np.reshape(np.sum(x_random*x_placeholders, axis=1), (size,1))
    
    # Creating data with dimensions (batch size, n_steps, n_input)
    data = np.dstack((x_random, x_placeholders))
    
    return data, y_labels

def make_D(theta):
    Dr = np.cos(theta) +1j*np.sin(theta)
    D  = np.diag(Dr)

    return Dr,D  


# Initialization of skew-Hermitian matrix A
def A_init(n_hidden):
    
    # Create real-symmetric component of A
    s = np.random.uniform(0, (2.0)*np.pi, \
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

# Used to update skew-hermision matrix A
def Update_step_A1(step_size,A,gradients,updatetype,t,s,r,si,ri):
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
    

# Defining RNN architecture
def RNN(x):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    
    
    # get rnn cell
    if model == 'LSTM':
        rnn_cell = rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
        rnnoutput = outputs[-1]
        
        # last layer, linear
        output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None)
    
    if model == 'ScuRNN':
        rnn_cell = scoRNNCell(n_input,n_hidden)
        state_init = tf.get_variable("ho", [1,2*n_hidden], \
                     initializer=init_ops.random_uniform_initializer(-0.01, 0.01))

        Initial_state = tf.tile(state_init, [batch_size, 1])
        
        # Place RNN cell into RNN, take last timestep as output
        outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=Initial_state, dtype=tf.float32)   #
        rnnoutput = outputs[-1]

        # Last layer, linear
        output = tf.layers.dense(inputs=rnnoutput, units=n_classes, activation=None,\
                             kernel_initializer=init_ops.glorot_uniform_initializer(seed=5544,dtype=tf.float32))
    
    
     
    return output
    


# Used for printing values
def getprintvalues(A, W):
    I = np.identity(A.shape[0])
    orthogonalcheck = np.linalg.norm(I - np.dot(np.conjugate(W).T,W))
    A_norm = np.linalg.norm(A, ord=1)
    IA_inverse = np.linalg.lstsq(I + A, I)
    IA_inverse_norm = np.linalg.norm(IA_inverse[0], ord=1)
    IW_norm = np.linalg.norm(I + W, ord=1)
    
    return orthogonalcheck, A_norm, IA_inverse_norm, IW_norm


def graphlosses(tr_mse, te_mse, xax):
    
    xax = [x/100000.0 for x in xax]
    
    plt.plot(xax, tr_mse, label='training MSE')
    plt.plot(xax, te_mse, label='testing MSE')
    plt.ylim([0,0.25])
    plt.legend(loc='lower left', prop={'size':6})
    plt.savefig(graphstring)
    plt.clf()
    
    return


# Generating training and test data
x_train, y_train = Generate_Data(training_size, n_steps)
#test_data, test_label = Generate_Data(testing_size, n_steps)
test_batches = [Generate_Data(50, n_steps) for i in range(200)]
#tlabel = np.split(test_label, 200)

with tf.Session() as sess:
    with tf.device('/cpu:0'):
        # tf Graph input
        x = tf.placeholder("float", [None, n_steps, n_input])
        y = tf.placeholder("float", [None, n_classes])
        
        # Assigning variable to RNN function
        pred = RNN(x)
        
        # Define loss object
        cost = tf.reduce_mean(tf.squared_difference(pred, y))
        
        
        # Variable lists
        Wvar = [v for v in tf.trainable_variables() if 'W:0' in v.name][0]
        Thetavar = [v for v in tf.trainable_variables() if 'T:0' in v.name][0]
        bvar = [v for v in tf.trainable_variables() if 'b:0' in v.name][0]
        Uvar = [v for v in tf.trainable_variables() if 'U:0' in v.name][0]
        Vvar = [v for v in tf.trainable_variables() if 'dense/kernel:0' in v.name][0]
        cvar = [v for v in tf.trainable_variables() if 'dense/bias:0' in v.name][0]
        hovar = [v for v in tf.trainable_variables() if 'ho:0' in v.name][0]
   
        
        
        
        # Define optimizer object
        optimizer_dict = {'adam' : tf.train.AdamOptimizer,
                          'adagrad' : tf.train.AdagradOptimizer,
                          'rmsprop' : tf.train.RMSPropOptimizer,
                          'sgd' : tf.train.GradientDescentOptimizer}
        
        opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)
        opt2 = optimizer_dict[Theta_optimizer](learning_rate=Theta_lr)
            
        scuRNNtrain = opt1.minimize(cost,var_list = [bvar,Uvar,Vvar,cvar,hovar])
        # Getting gradients
        grads_W = tf.gradients(cost,[Wvar])[0]
        #applygrad1 = opt1.minimize(cost, [othervarlist])
        
        # Updating variables
        newW = tf.placeholder(tf.float32, Wvar.get_shape())
        grad_Theta = tf.placeholder(tf.float32,Thetavar.get_shape())
        applygradtheta = opt2.apply_gradients([(grad_Theta, Thetavar)])
        updateW = tf.assign(Wvar, newW)

        # Initializing the variables
        init = tf.global_variables_initializer()
        
        # Launch the graph, tensorboard log, saver
        #datestring = time.strftime("%Y%m%d", time.gmtime())
        #if not os.path.exists("./savedmodels/" + datestring):
        #    os.makedirs("./savedmodels/" + datestring)
        sess.run(init)
        A =A_init(n_hidden)
        #print("skew-hermision :",A+(np.conjugate(A)).T)
        I = np.identity(A.shape[0])
        IplusAinv = np.linalg.lstsq(I+A,I)[0]
        IminusA = I-A
        theta = sess.run(Thetavar)
        Dr,D = make_D(theta)
        W = makeW(A,IplusAinv,D)
        W_expanded = Expand_Matrix(W)
        sess.run(updateW, feed_dict = {newW: W_expanded})
        # Keep training until reach number of epochs
        epoch = 1
        while epoch <= training_epochs:
            step = 1
            # Keep training until reach max iterations
            while step * batch_size <= training_size:
                        
                       
                # Getting input data
                batch_x = x_train[(step-1)*batch_size:step*batch_size,:,:]
                batch_y = y_train[(step-1)*batch_size:step*batch_size]   
            
                   
                # Updating input-output weights
                if model == 'LSTM':
                    sess.run(applygrad, feed_dict={x: batch_x, y: batch_y})
               

    		    # Updating hidden to hidden weights            
                if model == 'ScuRNN':
                    _, hidden_grads = sess.run([scuRNNtrain, grads_W], feed_dict = {x: batch_x, y: batch_y})
                    reduced_hidden_grads = Reduce_Matrix(hidden_grads)
                    DFA,DFT = Cayley_Transform_Deriv(reduced_hidden_grads, IplusAinv,IminusA, W,Dr,D) 
                    sess.run(applygradtheta, feed_dict = {grad_Theta: DFT})
                    theta = sess.run(Thetavar)
                    Dr,D = make_D(theta)
                    # Update A matrix by standard gradient descent  # Use other optimizers
                    A,tA,sA,rA,sAi,rAi = Update_step_A1(A_lr,A,DFA,A_optimizer,tA,sA,rA,sAi,rAi)
                    IplusAinv = np.linalg.lstsq(I+A,I)[0]
                    IminusA = I-A
                    W = makeW(A,IplusAinv,D)
                    W_expanded = Expand_Matrix(W)
                    sess.run(updateW, feed_dict = {newW: W_expanded})
                    
                        
                # Printing commands
                if step % display_step == 0:
                   
                    train_mse = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    test_mse = np.mean([sess.run(cost, feed_dict={x: test_data, y: test_label}) for test_data, test_label in test_batches])

                    # Evaluating average epoch accuracy/loss of model
                                    
                    displaystring = "Epoch " + str(epoch) + ", Iter " + \
                          str(step*batch_size) + ", Minibatch MSE= " + \
                          "{:.6f}".format(train_mse) + ", Test MSE= " + \
                          "{:.6f}".format(test_mse)
                        
                    if model == 'ScuRNN':

                        orthogonalcheck, A_norm, IA_inverse_norm, IW_norm = getprintvalues(A, W)
                        displaystring += ", Orthogonality Score= " + \
                            "{:.5f}".format(orthogonalcheck) + ", SkewA 1-Norm= " + \
                            "{:.5f}".format(A_norm) + ", (I+A)^-1 1-norm= " + \
                            "{:.5f}".format(IA_inverse_norm) + ", (I+W) 1-norm= " + "{:.5f}".format(IW_norm)
                    
                 
                    print(displaystring)
                        
                    # Plotting
                    number_iters_plt.append(step*batch_size + (epoch-1)*training_size)
                    train_mse_plt.append(train_mse)
                    test_mse_plt.append(test_mse)
                    orthogonal_plt.append(orthogonalcheck)
                    
                    graphlosses(train_mse_plt, test_mse_plt, number_iters_plt)

                    np.savetxt(savestring[:-4] + '._test_MSE.csv', test_mse_plt, delimiter = ',')
                    np.savetxt(savestring[:-4] + '._iters.csv', number_iters_plt, delimiter = ',')
                    np.savetxt(savestring[:-4] + '._orthogonal.csv', orthogonal_plt, delimiter = ',')
            
                step += 1
                
            # Calculate accuracy for test data
            epoch_mse = np.mean([sess.run(cost, feed_dict={x: test_data, y: test_label}) for test_data, test_label in test_batches])

            # Evaluating average epoch accuracy/loss of model
            print('\n')
            print("Testing MSE:", epoch_mse)
            print('\n')
              
            epoch += 1
            
            
            
        print("Optimization Finished!")
        
