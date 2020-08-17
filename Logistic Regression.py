import numpy as np
import math
import matplotlib.pyplot as plt
def gen_arr(path):
    return np.genfromtxt(path, delimiter=',')
def calc_lin(w,b,x):
    return np.dot(x,w.T)+b
def initialize_weights(x):
    w = np.zeros((1,x.shape[1]))
    return w,0
def sigmoid(z):
    denom = 1 + np.exp(-z)
    return 1/denom

def calc_cost(w,b,x,y):
    m =  x.shape[0]
    y_hat = sigmoid(calc_lin(w,b,x))
    cost = -1 * np.mean(np.multiply(y, np.log(y_hat)) + np.multiply(1.0 - y, np.log(1.0 - y_hat)))
    return cost

def gradient_descent(w,b,x,y,alpha):
    m = x.shape[0]
    y_hat = sigmoid(calc_lin(w,b,x))
    diff = y_hat-y
    db = np.sum(diff)/m
    dw = np.sum(np.dot(diff.T,x), axis =0)/m
    w -= alpha*dw
    b -= alpha*db
    cost = calc_cost(w,b,x,y)
    return w,b,cost
def optimize(w,b,x,y,alpha,num_iter):
    costs =list()
    iter =[]
    cost = calc_cost(w,b,x,y)
    for i in range(num_iter):
        if i % 100 == 0:
            costs.append(cost)
            iter.append(i)
        if i % 100 == 0:
            print('cost after iteration ' + str(i) + " :" + str(cost))
        w, b, cost = gradient_descent(w, b, x, y, alpha)
    return w,b,costs,iter

def predict(w,b,x,y):
    y_hat = sigmoid(calc_lin(w,b,x))
    y_hat = np.where(y_hat >= 0.5, 1, 0)
    percent = 100-(np.mean(np.abs(y_hat-y)))*100
    return percent

#y_hat = np.where(y_hat >= 0.5, 1, 0)
arr = gen_arr('ex2data1.txt')
x = arr[:,0:2]
y = arr[:,2].reshape(100,1)
w, b= initialize_weights(x)
pre = predict(w,b,x,y)
w,b,costs,iter = optimize(w,b,x,y,0.001,20000)
after = predict(w,b,x,y)
print(pre)
print(after)
plt.plot(iter,costs)
plt.show()