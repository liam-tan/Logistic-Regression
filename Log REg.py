import numpy as np


def initialize_weights(x):
    w = np.zeros((1, x.shape[1]))
    return w, 0


def gen_arr(path):
    return np.genfromtxt(path, delimiter=',')


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def propagate(w, b, X, Y):
    epsilon = 1e-5
    m = X.shape[0]
    z = np.matmul(X, w.T) + b
    a = sigmoid(z)
    J = -1 * np.mean(np.multiply(Y, np.log(a+epsilon)) + np.multiply(1.0 - Y, np.log(1.0 - a+epsilon)))
    dw = np.matmul((a - Y).T, X) * 1.0 / m
    db = np.mean(a - Y)
    cost = np.squeeze(J)
    grads = {'dw': dw, 'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iter, learning_rate, print_cost=False):
    costs = list()
    for i in range(num_iter):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print('cost after iteration ' + str(i) + " :" + str(cost))
    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[0]
    y_hat = np.zeros((m, 1))
    w = w.reshape(x.shape[1], 1)
    y_hat = sigmoid(np.matmul(X, w) + b)
    y_hat = np.where(y_hat >= 0.5, 1, 0)
    return y_hat


arr = gen_arr('ex2data1.txt')
x = arr[:, 0:2]
y = arr[:, 2].reshape(100, 1)

w, b = initialize_weights(x)


def model(x, y, num_iter=2000, learning_rate=0.5, print_cost=False):
    w,b = initialize_weights(x)
    params, grads, costs = optimize(w,b,x,y,num_iter,learning_rate,print_cost)
    w = params['w']
    b = params['b']

    y_pred = predict(w,b,x)
    percent = np.sum(np.abs(y_pred - y)) / y.shape[0]
    print(percent)

grads,cost=propagate(w, b, x, y)
print(grads)

