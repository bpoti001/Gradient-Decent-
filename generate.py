import random
import numpy as np
from sklearn import linear_model
import csv
from numpy import genfromtxt
#generating 3 weights and storing in list

w=[]

for i in range(3):
	w.append(random.uniform(0, 1)) #generates random float between 0 and 1 and appends to list w

w = np.asarray(w) #converting into array 
print ("inital weights" ,w)
#creating two vectors of 100000 rows which will act as features 
X1 = np.random.rand(100000,1) 
X2 = np.random.rand(100000,1)
#initalizing lables with zeros 100000 features will have 100000 lables. 
Y = np.zeros((100000,1))

#calclulating lables. 

Y = w[0]+w[1]*X1+w[2]*X2

#pairing up lable with its X1 and X2 to write into a file 
pair = []
for i in range(len(X1)):
	pair.append([X1[i],X2[i],Y[i]])
pair = np.asarray(pair)
trans = np.zeros((100000,3))

for i in range(len(pair)):
	trans[i]=pair[i].T
w = w.T
np.savetxt("generated_data_100000_1.csv", trans,fmt='%10.5f', delimiter=",")
np.savetxt("generated_weights_100000_1.csv", w,fmt='%10.5f', delimiter=",")
my_data = genfromtxt('generated_data_100000_1.csv', delimiter=',')
row = len(my_data)
coloum = len(my_data[0])

fetures = my_data[:,0:coloum-1]

lable = my_data[:,coloum-1]

lable = np.reshape(lable,(row,1))

clf = linear_model.LinearRegression()

clf.fit(fetures,lable)

print ("\n\n====checked-weights===\n\n")
print ("weights " ,clf.coef_)

print ("intercept" ,clf.intercept_)
