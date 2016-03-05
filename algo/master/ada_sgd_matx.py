import numpy as np
import random
import sys
import time
my_data = np.genfromtxt('/root/generated_data/generate/test/generated_data_1000000.csv', delimiter=',')
row = len(my_data)
coloum = len(my_data[0])
w = np.zeros((coloum))
w = w.reshape(coloum,1)
prediction = 0
alpha = 1e-2
fd = 1e-6
batch_size = 1 
batch_data=[]
win = [0.06724,0.08293,0.89167]
win = np.asarray(win)
win = win.reshape(3,1)
zer = np.zeros((3,1))
D_t = 0 
print "initial weights " ,w
alpha_i = 0
j = 0
#iteration = int(sys.argv[1])
new_coloum = np.ones((row,1))
data = np.append(new_coloum,my_data,1)
for a in xrange(0,row,batch_size):
   batch_data.append(data[a:a+batch_size])
while (np.isclose(w,win,atol=0.0000008).all()==False):
#for j in range(iteration):
   j = j+1
   for i in range(len(batch_data)):
      batch_array = batch_data[i]
      row_i = len(batch_array)
      coloum_i = len(batch_array[0])
      X = batch_array[:,0:coloum_i-1]
      Y = batch_array[:,[coloum_i-1]]
      prediction = np.dot(X,w)
      err = (prediction -Y)*X
      err_s = np.sum(err,axis=0)
      G_t = np.diag(err_s)
      D_t= D_t+(G_t*G_t)
      A = alpha/(fd + np.sqrt(D_t))
      gradient = np.multiply(A,G_t)
      w = w - gradient
      w = np.diag(w)
      w= w.reshape(3,1)
   print "error/slope",err_s
   print "added gradient",np.diag(D_t)
   print "in iteration" , j
   print "weight ",w
   err_c = err_s.reshape((3,1))
   print "isclose_error",np.isclose(err_c,zer,atol=0.0000041)
   print "isclose_weights",np.isclose(w,win,atol=0.0000008)
