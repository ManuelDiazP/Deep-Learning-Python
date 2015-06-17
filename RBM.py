import numpy as np
import csv
import pickle

import scipy as sp

from   scipy import optimize

from   activation import *

#-----------------------------------------------------------------------
  
import pickle
import random

#-----------------------------------------------------------------------

def save (x, filename):
    pkl_file = open(filename, 'wb')
    pickle.dump(x, pkl_file)
    pkl_file.close()
    print "NET_SAVED"

#-----------------------------------------------------------------------  

def load (filename):
    return pickle.load(open(filename, "rb" ))

#-----------------------------------------------------------------------
#version 0.2

class RBM:
  
  def __init__(self, num_visible, num_hidden, activation='sigmoid'):
      """
      This function initiates the RBM
      """

      self.type_model ='RBM' #Restricted Boltzmann Machine

      self.num_hidden   = num_hidden
      self.num_visible  = num_visible
      
      self.activation   = activation

      self.weights      = 0.1 * np.random.randn(num_visible, num_hidden)
      self.hidden_bias  = 0.1 * np.random.randn(num_hidden)
      self.visible_bias = 0.1 * np.random.randn(num_visible)
      
      # dw will hold last change in weights (for momentum)
      self.dw  = np.zeros((num_visible, num_hidden))
      self.dhb = np.zeros(num_hidden)
      self.dvb = np.zeros(num_visible)
      
      # Reset weights
      self.reset()

#-----------------------------------------------------------------------

  def reset (self):
      
      W_min = -0.5
      W_max =  0.5

      if (self.activation=='sigmoid'):

         W_min = -4*np.sqrt(6./(self.num_hidden+self.num_visible))
         W_max =  4*np.sqrt(6./(self.num_hidden+self.num_visible))
 
      if (self.activation=='tanh'):
    
         W_min = np.sqrt(6./(self.num_hidden+self.num_visible))
         W_max = np.sqrt(6./(self.num_hidden+self.num_visible))
    
      self.weights      = np.asarray(np.random.uniform(low=W_min,high=W_max,size=(self.num_visible, self.num_hidden)))
      self.hidden_bias  = np.zeros(self.num_hidden)
      self.visible_bias = np.zeros(self.num_visible)

#-----------------------------------------------------------------------

  def train(self, data, nepochs = 2500, lrate = 0.01, dlr = 0.001, momentum=0.01):
      """
      Train the RBM

      Parameters
      ----------
      data: A matrix where each row is a training example consisting of the states of visible units.    
      """

      factivation,dactivation  = activation_functions (self.activation)

      num_examples = data.shape[0]
    
      activation_bias = np.ones(num_examples)

      for epoch in range(nepochs):   

         pos_hidden_activations  = np.dot(data, self.weights)+self.hidden_bias    
         pos_hidden_probs        = factivation(pos_hidden_activations)
      
         #pos_hidden_states       = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden)
         pos_hidden_states       = pos_hidden_probs

         pos_associations_w      = np.dot(data.T         , pos_hidden_probs)#pos_hidden_states???
         pos_associations_vb     = np.dot(activation_bias, data)
         pos_associations_hb     = np.dot(activation_bias, pos_hidden_probs)

         neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)+self.visible_bias
         neg_visible_probs       = factivation(neg_visible_activations)
   
         neg_hidden_activations  = np.dot(neg_visible_probs, self.weights)+self.hidden_bias 
         neg_hidden_probs        = factivation(neg_hidden_activations)

         neg_associations_w      = np.dot(neg_visible_probs.T, neg_hidden_probs)
         neg_associations_vb     = np.dot(activation_bias, neg_visible_probs)
         neg_associations_hb     = np.dot(activation_bias, neg_hidden_probs)    
   
         delta_weights           =   lrate * ((pos_associations_w  - neg_associations_w)  / num_examples)    
         delta_hidden_bias       = 2*lrate * ((pos_associations_hb - neg_associations_hb) / num_examples)
         delta_visible_bias      = 2*lrate * ((pos_associations_vb - neg_associations_vb) / num_examples) 
   
         self.weights           += delta_weights      + momentum*self.dw
         self.hidden_bias       += delta_hidden_bias  + momentum*self.dhb 
         self.visible_bias      += delta_visible_bias + momentum*self.dvb
         
         self.dw  = delta_weights
         self.dhb = delta_hidden_bias
         self.dvb = delta_visible_bias

         error = np.sum((data - neg_visible_probs) ** 2)/(2*num_examples)
         
         lrate = lrate/(1+dlr)   
      
#-----------------------------------------------------------------------
      
  def prop_up (self, data):
      """
      This function propagates the visible units activation upwards to
      the hidden units
      """

      factivation,dactivation  = activation_functions (self.activation)

      return factivation(np.dot(data, self.weights)+self.hidden_bias)    

#-----------------------------------------------------------------------

  def prop_down (self, data):
      """
      This function propagates the hidden units activation downwards to
      the visible units
      """

      factivation,dactivation  = activation_functions (self.activation)

      return factivation(np.dot(data, self.weights.T)+self.visible_bias)

#-----------------------------------------------------------------------

  def error (self, data):
      """
      Autocoding error for data ...
      """
      
      num_examples = data.shape[0]
    
      z = self.prop_up(data)
      z = z > np.random.rand(num_examples, self.num_hidden)
      z = self.prop_down(z)
      error = np.sum((data - z) ** 2)
      return error
      
#-----------------------------------------------------------------------

  def rmse (self, data):
      """
      RMS error ...
      """

      z = self.prop_up(data)
      z = self.prop_down(z)
      error = np.sqrt(np.mean((data - z) ** 2))
      return error

# ----------------------------------------------------------------------

  def SGD(self, x, nruns=5, nsamples=10, nepoch=200, lrate=0.1, dlr = 0.001, momentum=0.1):
      """
      stochastic gradient descent
      """

      for i in range(nruns):

        idx = np.random.choice(np.arange(len(x)),nsamples,replace=False)

        xi=x[idx]

        print '%.8f' % self.rmse (xi),"->",

        self.train(xi,nepochs = nepoch, lrate = lrate, dlr = dlr, momentum=momentum)
        lrate=lrate/(1+dlr)
        
        print '%.8f' % self.rmse (xi)

#-----------------------------------------------------------------------

  def save (self, filename):
      save (self,filename)
      
#-----------------------------------------------------------------------

