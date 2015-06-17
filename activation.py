import numpy as np

#-----------------------------------------------------------------------

@np.vectorize
def fa_sigmoid(x):
    if (x<-500):
      return 0.0  
    return 1.0/(1.0+np.exp(-x))

def da_sigmoid(y):
    return y*(1-y)

#-----------------------------------------------------------------------

def fa_tanh(x):
    return np.tanh(x)

def da_tanh(y):
    return 1-(y*y)
    
#-----------------------------------------------------------------------

def fa_linear(x):
    return x

def da_linear(y):
    return 1
    
#-----------------------------------------------------------------------

@np.vectorize
def fa_rectified_linear(x):
    if (x<=0):
      return 0.0  
    return x

@np.vectorize
def da_rectified_linear(y):
    if (y<=0):
      return 0.0  
    return 1

#-----------------------------------------------------------------------

@np.vectorize
def fa_step(x):
    if (x<0):
      return 0.0  
    return 1.0

def da_step(y):
    return 1.0

#-----------------------------------------------------------------------

def activation_functions (activation):

    if (activation=='sigmoid'):
        return fa_sigmoid, da_sigmoid

    if (activation=='tanh'):
        return fa_tanh, da_tanh

    if (activation=='linear'):
        return fa_linear, da_linear
        
    if (activation=='rectified'):
        return fa_rectified_linear, da_rectified_linear

    if (activation=='step'):
        return fa_step, da_step
        
#-----------------------------------------------------------------------  

#gaussian
#f = exp(-(x*x)/2)
#F'=-x*exp(-(x*x)/2)

#-----------------------------------------------------------------------
