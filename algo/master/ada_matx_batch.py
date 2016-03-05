import numpy as np
import random
import sys
datasize =[100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
for j in datasize:
		my_data = np.genfromtxt('/root/generated_data/generate/test/generated_data_%s.csv'%(str(j)), delimiter=',')
		print "read data from generated_data_%s.csv"%(str(j))
		print "running for datasize ",j
		row = len(my_data)
		coloum = len(my_data[0])
		#w = np.zeros((coloum))
		#w = w.reshape(coloum,1)
		batch_size =[10,20,30,40,50,60,70,80,90,100]
		win = np.genfromtxt('/root/generated_data/generate/test/generated_weights_%s.csv'%(str(j)), delimiter=',')
		#win = np.asarray(win)
		win = win.reshape(3,1)
		zer = np.zeros((3,1))
		#print "initial weights " ,w
		#iteration = int(sys.argv[1])
		new_coloum = np.ones((row,1))
		data = np.append(new_coloum,my_data,1)
		for batch in batch_size:
			print "for batch_size",batch
			batch_data=[]
			alpha = 1e-2
			fd = 1e-6
			prediction = 0
			D_t = 0
			k = 0
			w = np.zeros((coloum))
			w_old = np.ones((coloum))
			w_old = w_old.reshape(coloum,1)
			w = w.reshape(coloum,1)
			for a in xrange(0,row,batch):
				batch_data.append(data[a:a+batch])
			random.shuffle(batch_data)
			#while ((np.isclose(w,win,atol=0.0000008).all()==False)or((w!=w_old).all())):
			while ((w!=w_old).all()):
				k = k+1
				w_old = w
				print "w_old",w_old
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
				#print "error/slope",err_s
				#print "added gradient",np.diag(D_t)
				print "in iteration" , k
				print "weight ",w
				#w_old = w
				#err_c = err_s.reshape((3,1))
				#print "check",np.isclose(w,w_old,atol=0).all()
				#print "isclose_error",np.isclose(err_c,zer,atol=0.0000041)
				#print "isclose_weights",np.isclose(w,win,atol=0.0000008)
			fi = open ('/root/generated_data/output_adagrad/data_'+str(j)+'/output_batch_'+str(batch)+'.txt', 'w')
			value_1 = ("batch_size ", batch)
			value_2 = ("itrations",k)
			value_3 = ("generated weights",w)
			val = str(value_1)+'\n'+str(value_2)+'\n'+str(value_3)
			print val
			fi.write(str(val))
			fi.close()
