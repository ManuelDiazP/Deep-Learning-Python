# -*- encoding: utf-8 -*-
#-----------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy  as np
import pandas as pd
import scipy  as sp
import os

from   pylab import imread, imshow, gray, mean
from   sklearn.feature_extraction import DictVectorizer

import RBM as rbm
import MLP as mlp

#-----------------------------------------------------------------------
  
import pickle
import random

#-----------------------------------------------------------------------

def save (x, filename):
    pkl_file = open(filename, 'wb')
    pickle.dump(x, pkl_file)
    pkl_file.close()

#-----------------------------------------------------------------------  

def load (filename):
    return pickle.load(open(filename, "rb" ))

#-----------------------------------------------------------------------
#IMAGE VECTORIZATION
#-----------------------------------------------------------------------

def img2vector (filename):

         x = imread(filename)
       
         #z = sp.misc.imresize(x, 1.0, interp='bilinear', mode=None)

         r = x[140:284,140:284,0]
         r = sp.misc.imresize(r, (10,10), interp='bilinear', mode=None)
         r=(r.astype(float)/255)
         
         g = x[140:284,140:284,1]
         g = sp.misc.imresize(g, (10,10), interp='bilinear', mode=None)
         g =(g.astype(float)/255)
         
         b = x[140:284,140:284,2]
         b = sp.misc.imresize(b, (10,10), interp='bilinear', mode=None)
         b =(b.astype(float)/255)
 
         glx=np.concatenate((r,g,b))

         glx=np.reshape(glx,300)
       
         return glx

#-----------------------------------------------------------------------
#RBM01
#-----------------------------------------------------------------------

def train_rbm_01():

    df = pd.read_csv('DATA/training_solutions_rev1.csv')

    #rbm_01=rbm.RBM(1200,50,activation='sigmoid')
    rbm_01=load ("NET/rbm_01.net")

    for k in range(2):#REPITE 50 VECES

      rows = random.sample(df.index, 250)#MUESTREA 1000 IMAGENES DE ENTRENAMIENTO
      df_1000 = df.ix[rows]

      #imagen= df_1000['']

      files = ['DATA/images_training_rev1/'+str(imagen)+'.jpg' for imagen in df_1000['GalaxyID']]

      filas=[]#VECTORIZA IMAGENES

      for filename in files:
         filas.append(img2vector(filename))
         print ".",
         
      print "."
      print "PROP UP"

      x=np.array(filas)
      
      print "x=",x.shape

      print "TRAIN RBM 01"
      rbm_01.SGD(x, nruns=10, nsamples=25, nepoch=50, lrate=0.01, dlr = 0.0, momentum=0.5)
      
    save (rbm_01, "NET/rbm_01.net")

#-----------------------------------------------------------------------
#RBM02
#-----------------------------------------------------------------------

def train_rbm_02():

    df = pd.read_csv('DATA/training_solutions_rev1.csv')

    #rbm_02=rbm.RBM(37,10,activation='sigmoid')
    rbm_02=load ("NET/rbm_02.net")

    for k in range(2):#REPITE 50 VECES

      rows = random.sample(df.index, 1000)#MUESTREA 1000 IMAGENES DE ENTRENAMIENTO
      df_1000 = df.ix[rows]
      
      y = df_1000.as_matrix(columns=None)[:,1:].astype(np.float32, copy=False)

      print "TRAIN RBM 02"
      
      rbm_02.SGD(y, nruns=250, nsamples=5, nepoch=50, lrate=0.01, dlr = 0.0, momentum=0.5)
      
    save (rbm_02, "NET/rbm_02.net")

    for k in range(2):#REPITE 50 VECES

      rows = random.sample(df.index, 1000)#MUESTREA 1000 IMAGENES DE ENTRENAMIENTO
      df_1000 = df.ix[rows]
      
      y = df_1000.as_matrix(columns=None)[:,1:].astype(np.float32, copy=False)

      print "TRAIN RBM 02"
      
      rbm_02.SGD(y, nruns=250, nsamples=5, nepoch=50, lrate=0.001, dlr = 0.0, momentum=0.5)
      
    save (rbm_02, "NET/rbm_02.net")

#-----------------------------------------------------------------------
#RBM03
#-----------------------------------------------------------------------

def train_rbm_03():

    df = pd.read_csv('DATA/training_solutions_rev1.csv')

    rbm_01=load ("NET/rbm_01.net")
    rbm_02=load ("NET/rbm_02.net")
    
    #rbm_03=rbm.RBM(60,200,activation='sigmoid')
    rbm_03=load ("NET/rbm_03.net")
    
    df = pd.read_csv('DATA/training_solutions_rev1.csv')

    for k in range(2):#REPITE 50 VECES

      rows = random.sample(df.index, 250)#MUESTREA 1000 IMAGENES DE ENTRENAMIENTO
      df_1000 = df.ix[rows]
      
      y = df_1000.as_matrix(columns=None)[:,1:].astype(np.float32, copy=False)

      files = ['DATA/images_training_rev1/'+str(imagen)+'.jpg' for imagen in df_1000['GalaxyID']]

      filas=[]#VECTORIZA IMAGENES

      for filename in files:
         filas.append(img2vector (filename))
         print ".",
         
      print "."
      print "PROP UP"
         
      x=rbm_01.prop_up(np.array(filas))
      y=rbm_02.prop_up(y)
      
      z=np.hstack((x,y))
      
      print "z=",z.shape

      print "TRAIN RBM 03"
      
      rbm_03.SGD(z, nruns=10, nsamples=25, nepoch=50, lrate=0.1, dlr = 0.0, momentum=0.5)
      
    save (rbm_03, "NET/rbm_03.net")

#-----------------------------------------------------------------------
#MLP
#-----------------------------------------------------------------------

def gen_mlp():
    
    rbm_01=load ("NET/rbm_01.net")
    rbm_02=load ("NET/rbm_02.net")
    rbm_03=load ("NET/rbm_03.net")
    
    print  rbm_03.weights.shape
    
    net = mlp.MLP([300,50,200,50,37], activation='sigmoid')

    net.weights = [rbm_01.weights    ,rbm_03.weights[:50,:],rbm_03.weights.T [:,50:],rbm_02.weights.T]
    net.bias    = [rbm_01.hidden_bias,rbm_03.hidden_bias   ,rbm_03.visible_bias[50:],rbm_02.visible_bias]    

    save (net, "NET/mlp.net")
    
#-----------------------------------------------------------------------

def mlp2rbn():
    
    rbm_01=load ("NET/rbm_01.net")
    rbm_02=load ("NET/rbm_02.net")
    rbm_03=load ("NET/rbm_03.net")
    
    net=load ("NET/mlp.net")

    rbm_01.weights          =net.weights[0]
    rbm_03.weights [:50,:]  =net.weights[1]
    rbm_03.weights [50:,:]  =net.weights[2].T
    rbm_02.weights          =net.weights[3].T

    rbm_01.hidden_bias      =net.bias[0]
    rbm_03.hidden_bias      =net.bias[1]
    rbm_03.visible_bias[50:]=net.bias[2]
    rbm_02.visible_bias     =net.bias[3]

    save (rbm_01, "NET/rbm_01.net")
    save (rbm_02, "NET/rbm_02.net")
    save (rbm_03, "NET/rbm_03.net")

#-----------------------------------------------------------------------

def train_mlp():

    net=load ("NET/mlp.net")

    df = pd.read_csv('DATA/training_solutions_rev1.csv')

    for k in range(50):#REPITE 100 VECES

      rows = random.sample(df.index, 100)#MUESTREA 1000 IMAGENES DE ENTRENAMIENTO
      df_1000 = df.ix[rows]
      
      y = df_1000.as_matrix(columns=None)[:,1:].astype(np.float32, copy=False)

      files = ['DATA/images_training_rev1/'+str(imagen)+'.jpg' for imagen in df_1000['GalaxyID']]

      filas=[]#VECTORIZA IMAGENES

      for filename in files:
         filas.append(img2vector (filename))
         print ".",
         
      print "."
      print "PROP UP"
         
      x=np.array(filas)

      print "TRAIN MLP"
      
      net.SGD(x, y, nruns=1000, nsamples=1, nepoch=50, lrate=0.1, dlr = 0.0, momentum=0.0)
    
    save (net, "NET/mlp.net")
    
#----------------------------------------------------------------------- 

def submission():
    
    import csv
    
    c = csv.writer(open("MYFILE.csv", "wb"))
    
    net=load ("NET/mlp.net")

    df = pd.read_csv('DATA/all_zeros_benchmark.csv')
    #files = ['DATA/images_test_rev1/'+str(imagen)+'.jpg' for imagen in df['GalaxyID']]
    
    #filas=[]#VECTORIZA IMAGENES

    i=0
    for ident in df['GalaxyID']:
        filename='DATA/images_test_rev1/'+str(ident)+'.jpg'
        x=img2vector(filename)
        y=net.propagate_forward(x)
        c.writerow(y)
        print i
        i=i+1

#-----------------------------------------------------------------------  

def main():
    """
    np.random.seed(587482)
    
    rbm_01=rbm.RBM(300,50,activation='sigmoid')
    rbm_02=rbm.RBM(37,50,activation='sigmoid')
    rbm_03=rbm.RBM(100,200,activation='sigmoid')
    
    save (rbm_01, "NET/rbm_01.net")
    save (rbm_02, "NET/rbm_02.net")
    save (rbm_03, "NET/rbm_03.net")
    
    gen_mlp()
    
    train_rbm_01 ()
    train_rbm_02 ()
    train_rbm_03 ()
    
    gen_mlp      ()
    train_mlp    ()
    mlp2rbn()
        
    for k in range(100):

      train_rbm_01 ()
      gen_mlp      ()
      train_mlp    ()
      print "N=",k
      mlp2rbn()
      
      train_rbm_02 ()
      gen_mlp      ()
      train_mlp    ()
      print "N=",k
      mlp2rbn()
            
      train_rbm_03 ()
      gen_mlp      ()
      train_mlp    ()
      print "N=",k
      mlp2rbn()

    submission     ()
    """
    
    np.random.seed(587482)
    
    for k in range(10):

      train_mlp ()
      
      print "----------------------------------------------------------"
      
    submission     ()
    
#-----------------------------------------------------------------------  
    
if __name__=="__main__":
    main()
 
#-----------------------------------------------------------------------  
