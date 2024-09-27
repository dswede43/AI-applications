#!/usr/bin/python3

#Gradient descent functions
#---
#Functions required to run the gradient descent algorithm to build a 2-layer neural network.


#Table of contents:
#---
#1. Forward propagation
#2. Back propagation
#3. Update parameters
#4. Gradient descent algorithm


#load libraries
import numpy as np


#1. Forward propagation
#---
#initialize model parameters
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

#define the ReLU activation function
def ReLU(Z):
    return np.maximum(0, Z)

#define the sigmoid function
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#define the forward propagation function
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

#define one-hot encoding function
def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


#2. Back propagation
#---
#define the derivative of ReLU
def dReLU(Z):
    return Z > 0

#define the backward propagation function
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2


#3. Update parameters
#---
#function to update parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


#4. Gradient descent algorithm
#---
#function to return model predictions
def predict(A2):
    return np.argmax(A2, 0)

#function to make predictions given the model parameters
def make_prediction(W1, b1, W2, b2, X):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    y_pred = predict(A2)
    return y_pred

#function to calculate model accuracy
def get_accuracy(y_pred, Y):
    return np.sum(y_pred == Y) / Y.size

#function to calculate model loss/error
def get_loss(A2, Y):
    one_hot_Y = one_hot_encode(Y)
    return -np.sum(one_hot_Y * np.log(A2)) / Y.size

#define gradient descent algorithm
def gradient_descent(X, Y, iterations, alpha):
    #define an empty dictionary
    train_results = {}

    #initialize random parameter estimates
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        #forward propagation
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        #backward propagation
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)

        #update parameters
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        #calculate the model accuracy
        y_pred = predict(A2)
        accuracy = get_accuracy(y_pred, Y)
        loss = get_loss(A2, Y)

        #every 10th iteration
        if (i % 10 == 0):
            print(f"Iteration: {i}")
            print(f"Loss: {loss}")
            print(f"Accuracy: {accuracy}")

            #store results
            train_results[i] = [loss, accuracy]
    return W1, b1, W2, b2, train_results
