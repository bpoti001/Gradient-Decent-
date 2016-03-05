import numpy as np
import random
import sys
import time
my_data = np.genfromtxt('/root/generated_data/generate/test/generated_data_1000000.csv', delimiter=',')
row = len(my_data)
coloum = len(my_data[0])
#random.shuffle(my_data)
w = np.zeros((coloum))
w = w.reshape(coloum,1)
#alpha = 1
prediction = 0
alpha = 1e-2
#prediction = 0
d1 = 0
d2 = 0
d3 = 0
a1 = 0
a2 = 0
a3 = 0
fd = 1e-6
batch_size = 100 
batch_data=[]
#win = [0.30901,0.39808,0.16489]
#win = np.asarray(win)
#win = win.reshape(3,1)
print "initial weights " ,w
alpha_i = 0 
count = 0
#j = 0 
iteration = int(sys.argv[1])
for a in xrange(0,row,batch_size):
	batch_data.append(my_data[a:a+batch_size])
#while (np.isclose(w,win,atol=0.0000008).all()==False):

for j in range(iteration):
	#j = j+1
	for  i in range(len(batch_data)):
		count = count+1
    		batch_array = batch_data[i]
    		#random.shuffle(batch_array)
    		row_i = len(batch_array)
    		coloum_i = len(batch_array[0])
		fetures = batch_array[:,0:coloum_i-1]
		Y = batch_array[:,[coloum_i-1]]
		new_coloum = np.ones((row_i,1))
		X = np.append(new_coloum,fetures,1)
		#alpha_i = alpha/(row*np.sqrt(j))
		prediction = np.dot(X,w)
		#print "predictions" ,prediction
		error_x1 = (prediction -Y)*X[:,[0]]
		error_x2 = (prediction -Y)*X[:,[1]]
		error_x3 = (prediction -Y)*X[:,[2]]
		d1 = d1+ error_x1.sum()*error_x1.sum()
                d2 = d2+ error_x2.sum()*error_x2.sum()
                d3 = d3+ error_x3.sum()*error_x3.sum()
                a1 = alpha/(fd + np.sqrt(d1))
                a2 = alpha/(fd + np.sqrt(d2))
                a3 = alpha/(fd + np.sqrt(d3))
		##print "error\n"
		##print error_x1.sum()
		##print error_x2.sum()
		##print error_x3.sum()
		w[0] = w[0] - a1 *error_x1.sum()
		w[1] = w[1] - a2 *error_x2.sum()
		w[2] = w[2] - a3 *error_x3.sum()
		##print "updated weight",w
		#time.sleep(20)
	print "in iteration" , j
	print "weight ",w
	print "slope in d1", error_x1.sum()
	print "slope  in d2", error_x2.sum()
	print "slope in d3", error_x3.sum()
	#print "isclose",np.isclose(w,win,atol=0.0000008)
	#print count
#print "for batch_size ", batch_size 
#print "for iteration ", iteration
#print "final weight",w
#print "abs difference", abs(w-win)
#print error_x1.sum()
#print error_x2.sum()
#print error_x3.sum()
#print alpha_i
