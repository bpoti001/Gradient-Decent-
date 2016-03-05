import numpy as np
import random
import sys
my_data = np.genfromtxt('/root/generated_data/generated_data.csv', delimiter=',')
row = len(my_data)
coloum = len(my_data[0])
random.shuffle(my_data)
w = np.zeros((coloum))
w = w.reshape(coloum,1)
win = [0.73057,0.44719,0.80003]
win = np.asarray(win)
win =win.reshape(coloum,1)
alpha = 1e-2
prediction = 0
d1 = 0 
d2 = 0 
d3 = 0 
a1 = 0 
a2 = 0 
a3 = 0 
fd = 1e-6
#batch_size = 20
batch_data=[]
iteration = 0
#error_x1=0
#error_x2=0
#error_x3=0
print w
print win
batch_list = [20]
for a in batch_list:
	for i in xrange(0,row,a):
		batch_data.append(my_data[i:i+a])
	#while (np.isclose(w,win,atol=5.2e-06).all()==False):
	for a in range(sys.args[1])
		for  i in range(len(batch_data)):
    			batch_array = batch_data[i]
    			random.shuffle(batch_array)
    			row_i = len(batch_array)
    			coloum_i = len(batch_array[0])
			fetures = batch_array[:,0:coloum_i-1]
			Y = batch_array[:,[coloum_i-1]]
			new_coloum = np.ones((row_i,1))
			X = np.append(new_coloum,fetures,1)
			#lpha_i = alpha/(row*np.sqrt(i+1))
			prediction = np.dot(X,w)
			error_x1 = (prediction -Y)*X[:,[0]]
			error_x2 = (prediction -Y)*X[:,[1]]
			error_x3 = (prediction -Y)*X[:,[2]]
			d1 = d1+ error_x1.sum()*error_x1.sum()
			d2 = d2+ error_x2.sum()*error_x2.sum()
			d3 = d3+ error_x3.sum()*error_x3.sum()
			a1 = alpha/(fd + np.sqrt(d1))
			a2 = alpha/(fd + np.sqrt(d2))
			a3 = alpha/(fd + np.sqrt(d3))

			w[0] = w[0] - a1*error_x1.sum()
			w[1] = w[1] - a2*error_x2.sum()
			w[2] = w[2] - a3*error_x3.sum()
			#print "for batch " ,i,w
		iteration = iteration+1
	
	f = open('/root/generated_data/output/data_10000_1/output_batch_%s.txt' %(str(a)), 'w')
	value_1 = ( "for batch_size ", a) 
	value_1 = str(value_1)
	#f.write(str(value))
	print "total iterations", iteration
	print "for batch_size",a
	value_2= ("total iterations ", iteration)
	value_2 = str(value_2)
	print "weights", w
	value_3 = ("weights", w)
	value_3 = str(value_3)
	value = str(value_1+'\n'+value_2+'\n'+value_3)
	f.write(str(value))
	iteration=0
	w = np.zeros((coloum))
	w = w.reshape(coloum,1)
	alpha = 1
	prediction = 0
	alpha_i= 0
	batch_data=[]
#print "slope in d1",error_x1.sum()
#print "slope in d2",error_x2.sum()
#print "slope in d3",error_x3.sum()
#print "alpha value",alpha_i
