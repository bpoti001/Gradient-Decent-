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
alpha = 1
prediction = 0
#batch_size = 20
batch_data=[]
iteration = 0
#error_x1=0
#error_x2=0
#error_x3=0
alpha_i= 0
print w
print win
batch_list = [10,20,30,40,50,60,70,80,90,100]
for a in batch_list:
	for i in xrange(0,row,a):
		batch_data.append(my_data[i:i+a])
	while (np.isclose(w,win,atol=5.2e-06).all()==False):
		for  i in range(len(batch_data)):
    			batch_array = batch_data[i]
    			random.shuffle(batch_array)
    			row_i = len(batch_array)
    			coloum_i = len(batch_array[0])
			fetures = batch_array[:,0:coloum_i-1]
			Y = batch_array[:,[coloum_i-1]]
			new_coloum = np.ones((row_i,1))
			X = np.append(new_coloum,fetures,1)
			alpha_i = alpha/(row*np.sqrt(i+1))
			prediction = np.dot(X,w)
			error_x1 = (prediction -Y)*X[:,[0]]
			error_x2 = (prediction -Y)*X[:,[1]]
			error_x3 = (prediction -Y)*X[:,[2]]
			w[0] = w[0] - alpha_i *error_x1.sum()
			w[1] = w[1] - alpha_i *error_x2.sum()
			w[2] = w[2] - alpha_i *error_x3.sum()
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
