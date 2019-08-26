from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels =  len(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train.T
y_test = y_test.T

x_train = x_train.astype('float32')/255
x_test  = x_test.astype('float32')/255
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])

x_train = np.reshape(x_train, [60000,-1])
x_test = np.reshape(x_test, [10000,-1])
x_train = x_train.T
x_test  = x_test.T


np.random.seed(1)
fe = x_train
y = y_train
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def deriv_sigmoid(x):
    a = sigmoid(x)
    return a*(1-a)
prev = 0
next  =0
w0 = np.random.randn(128, fe.shape[0])
b0 = np.zeros((128, 1))
w1 = np.random.randn(256, 128)
b1 = np.zeros((256, 1))
w2 = np.random.randn(64, 256)
b2 = np.zeros((64, 1))
w3 = np.random.randn(10,64)
b3 = np.zeros((10,1))

for i in range(4000):
    z0 = w0.dot(fe)+b0
    a0 = sigmoid(z0)
    z1 = w1.dot(z0)+b1
    a1= sigmoid(z1)
    z2 = w2.dot(a1)+b2
    a2 = sigmoid(z2)
    z3 = w3.dot(a2)+b3
    a3 = sigmoid(z3)
    if i%1==0:
        loss = -np.sum(y*np.log(a3)+(1-y)*np.log(1-a3))
        print(loss)
    
    dz3 = a3 - y
    db3 = np.sum(dz3,axis = 1,keepdims = True)/fe.shape[0]
    dw3 = dz3.dot(a2.T)/fe.shape[0]
    
    dz2 = (w3.T.dot(dz3))*(deriv_sigmoid(z2))
    db2 = np.sum(dz2,axis = 1,keepdims = True)/fe.shape[0]
    dw2 = dz2.dot(a1.T)/fe.shape[0]
    
    
                       
    dz1 = (w2.T.dot(dz2))*deriv_sigmoid(z1)
    db1 = np.sum(dz1, axis=1,keepdims=True)/fe.shape[0]
    dw1 = dz1.dot(a0.T)/fe.shape[0]
    
    dz0 = (w1.T.dot(dz1))*deriv_sigmoid(z0)
    db0 = np.sum(dz0, axis=1,keepdims=True)/fe.shape[0]
    dw0 = dz0.dot(fe.T)/fe.shape[0]
    lr = 0.05
    
    w0 = w0 - lr * dw0
    b0 = b0 - lr * db0                       
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    w3 = w3 - lr * dw3
    b3 = b3 - lr * db3


q0 = sigmoid(w0.dot(x_test)+b0)
q1 = sigmoid(w1.dot(q0)+b1)
q2 = sigmoid(w2.dot(q1)+b2)
q3 = sigmoid(w3.dot(q2)+b3)

q3 = q3.T

y_test = y_test.T

pred = []
for j in range(len(q3)):
    pred.append(np.argmax(q3[j]))

actual = []
for j in range(len(y_test)):
    actual.append(np.argmax(y_test[j]))

true = 0
false = 0
for j in range(len(y_test)):
    if(actual[j]==pred[j]):
        true+=1
    else:
        false+=1
        
print("accuracy = ",true*100/(true+false),"%")
