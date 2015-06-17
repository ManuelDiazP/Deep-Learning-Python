#!/usr/bin/env python
# ----------------------------------------------------------------------
# Multi-layer perceptron
# ----------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with
# retropropagation learning.
# ----------------------------------------------------------------------

import numpy as np

from activation import *

#-----------------------------------------------------------------------
  
import pickle
import random

#-----------------------------------------------------------------------

class MLP:
  ''' Multi-layer perceptron class. '''

# ----------------------------------------------------------------------

  def __init__(self, dim , activation='sigmoid'):
        ''' Initialization of the perceptron with given sizes.  '''

        self.type_model ='MLP' #Multi Layer Perceptron
        
        self.activation=activation
        
        self.dim = dim
        
        n = len(dim)

        # Build weights matrix (randomly between -0.25 and +0.25)

        self.weights = []
        self.bias    = []

        for i in range(n-1):
            self.weights.append(np.zeros((dim[i], dim[i+1])))
            self.bias.append(np.zeros((dim[i+1])))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)
        self.db = [0,]*len(self.bias)

        # Reset weights
        self.reset()

# ----------------------------------------------------------------------

  def reset(self):
      ''' Reset weights '''

      for i in range(len(self.weights)):

        nw1=self.dim[i]
        nw2=self.dim[i+1]
        
        W_min = -0.5
        W_max =  0.5

        if (self.activation=='sigmoid'):

          W_min = -4*np.sqrt(6./(nw1+nw2))
          W_max =  4*np.sqrt(6./(nw1+nw2))
 
        if (self.activation=='tanh'):
    
          W_min = np.sqrt(6./(nw1+nw2))
          W_max = np.sqrt(6./(nw1+nw2))
              
        self.weights[i][...] = np.asarray (np.random.uniform(low=W_min,high=W_max,size=(nw1,nw2)))
        self.bias[i]         = np.random.random(nw2)
        
      print "reset"

# ----------------------------------------------------------------------

  def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''
        
        factivation,dactivation  = activation_functions (self.activation)

        # Imput layer
        layer = data

        for i in range(1,len(self.dim)):
            # Propagate activity
            layer=factivation(np.dot(layer,self.weights[i-1])+self.bias[i-1])

        # Return output
        return layer

# ----------------------------------------------------------------------

  def train (self, data, target, nepochs = 2500, lrate=0.1, dlr = 0.001, momentum=0.1):
        ''' Back propagate error. '''
        
        num_examples = target.shape[0]
        
        factivation,dactivation  = activation_functions (self.activation)
        
        for epoch in range(nepochs):
            
          # Propagate from layer 0 to layer n-1 using sigmoid as activation function
          layers=[data]

          # Propagate from layer 0 to layer n-1 using sigmoid as activation function
          for i in range(1,len(self.dim)):
            layers.append(factivation(np.dot(layers[-1],self.weights[i-1])+self.bias[i-1]))

          # Compute error on output layer
          error = (target - layers[-1])

          # Compute deltas
          deltas = [error*dactivation(layers[-1])]

          # Compute error on hidden layers
          for i in range(len(self.dim)-2,0,-1):
            deltas.insert(0,np.dot(deltas[0],self.weights[i].T)*dactivation(layers[i]))

          # Update weights
          for i in range(len(self.weights)):
            
            layer = np.atleast_2d(layers[i])

            dw = np.dot(layer.T,np.atleast_2d(deltas[i]))
            db = deltas[i].mean(0)

            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.bias[i]    += lrate*db + momentum*self.db[i]

            self.dw[i] = dw
            self.db[i] = db
            
          # Return error
          return np.sum(error**2)/(2*num_examples)

# ----------------------------------------------------------------------

  def SGD(self, x, y, nruns=200, nsamples=10, nepoch=200, lrate=0.1, dlr = 0.001, momentum=0.1):
      """
      stochastic gradient descent
      """
      
      for i in range(nruns):
        
        idx = np.random.choice(np.arange(len(x)),nsamples,replace=False)
        
        xi=x[idx]
        yi=y[idx]
        
        print '%.8f' % self.rmse (xi,yi),"->",
        
        self.train(xi,yi, nepoch, lrate, momentum)
        lrate=lrate/(1+dlr)
        
        print '%.8f' % self.rmse (xi,yi)

#-----------------------------------------------------------------------

  def rmse (self, x,y):
      """
      RMS error ...
      """

      z = self.propagate_forward(x)
      error = np.sqrt(np.mean((y - z) ** 2))
      return error
      
#-----------------------------------------------------------------------

  def save (self, filename):
      save (self,filename)
      
#-----------------------------------------------------------------------
"""
net = MLP([2,4,2], activation='sigmoid')

print "Learning the OR logical function"

net.reset()

x=np.array([[0,0],[1,0],[0,1],[1,1]])
y=np.array([[0,1],[1,0],[1,0],[1,0]])

# ----------------------------------------------------------------------

print "TEST:"
z=net.propagate_forward(x)
print z

# ----------------------------------------------------------------------

net.SCG(x,y, na=2000,nb=10, n=4, lrate=0.1, momentum=0.1)
    
# ----------------------------------------------------------------------

print "SOL:"
z=net.propagate_forward(x)
print z
"""
# ----------------------------------------------------------------------
# Learning sin(x)
# ----------------------------------------------------------------------
"""
import matplotlib
import matplotlib.pyplot as plt
    
print "Learning the sin function"
net = MLP([1,10,1], activation='tanh')
    
x = np.linspace(0.0,1.0,500).reshape((500,1))
y = np.sin(x*np.pi).reshape((500,1))

print "Learning..."

net.SCG(x, y, na=200000, nb=10, n=5, lrate=0.1, momentum=0.1)

# ----------------------------------------------------------------------

plt.figure(figsize=(10,5))

# Draw real function

plt.plot(x,y,color='b',lw=1)

# Draw network approximated function

z= net.propagate_forward(x)
plt.plot(x,z,color='r',lw=3)
plt.axis([0,1,0,1])
plt.show()

# ----------------------------------------------------------------------


  def propagate_backward(self, data, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error. '''

        deltas_w = []
        deltas_b = []
        
        # Compute error on output layer
        
        error = target - self.propagate_forward(data)#self.layers[-1]

        deltas_w.append(error*self.dactivation(self.layers[-1]))
        deltas_b.append((error).sum(0))
        
        k=target.shape[0]
        
        # Compute error on hidden layers
        for i in range(len(self.dim)-2,0,-1):

            deltas_w.insert(0,np.dot(deltas_w[0],self.weights[i].T)*self.dactivation(self.layers[i]))
            deltas_b.insert(0,deltas_b[0]*self.bias[i])
            
        # Update weights
        for i in range(len(self.weights)):
            
            layer = np.atleast_2d(self.layers[i])

            dw = np.dot(layer.T,np.atleast_2d(deltas_w[i]))
            db = deltas_b[i]/k
          
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.bias[i]    += lrate*db + momentum*self.db[i]
            
            self.dw[i] = dw
            self.db[i] = db
            
        # Return error
        return  (error**2).sum()






Following the notation of Rojas 1996, chapter 7, backpropagation 
computes partial derivatives of the error function E (aka cost, aka 
loss)

dE/dw[i,j] = delta[j] * o[i]
where w[i,j] is the weight of the connection between neurons i and j, j 
being one layer higher in the network than i, and o[i] is the output 
(activation) of i (in the case of the "input layer", that's just the 
value of feature i in the training sample under consideration). 

How to determine delta is given in any textbook and depends on the 
activation function, so I won't repeat it here.

These values can then be used in weight updates, e.g.

// update rule for vanilla online gradient descent
w[i,j] -= gamma * o[i] * delta[j]
where gamma is the learning rate.

The rule for bias weights is very similar, except that there's no input from a previous layer. Instead, bias is (conceptually) caused by input from a neuron with a fixed activation of 1. So, the update rule for bias weights is

bias[j] -= gamma_bias * 1 * delta[j]
where bias[j] is the weight of the bias on neuron j, the multiplication with 1 can obviously be omitted, and gamma_bias may be set to gamma or to a different value. IIRC, lower values are preferred, though I'm not sure about the theoretical justification of that.










"""
# ----------------------------------------------------------------------
