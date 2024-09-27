#!/usr/bin/python3

#Neural Network (NN) training animation
#---
#Script to animate the NN training.


#Table of contents:
#---
#1. Define training dataset
#2. Model animation


#load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
import gradient_descent_functions as grad_desc

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/NN from scratch" #working directory

#load data
train_data = pd.read_csv(f"{DIR}/data/mnist_train.csv")


#1. Define training dataset
#---
#subset the training data
np.random.seed(42)
train_data = train_data.sample(1000)

#define feature and target variables
m, n = train_data.shape

train_data = np.array(train_data).T
y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train / 255.0 #normalize data


#2. Model animation
#---
#function to apply gradient descent and animate the model training
def gradient_descent_with_animation(X, Y, iterations, alpha):
    #initialize random parameters
    W1, b1, W2, b2 = grad_desc.init_params()

    #calculate the PC scores for the training data
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X.T).T
    var = pca.explained_variance_ratio_

    #define the plot ranges
    x_min = np.min(X_pca[0, :])
    x_max = np.max(X_pca[0, :])
    y_min = np.min(X_pca[1, :])
    y_max = np.max(X_pca[1, :])

    #create the plot
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 - {var[0] * 100:.2f}%")
    ax.set_ylabel(f"PC1 - {var[1] * 100:.2f}%")

    #funtion to apply gradient descent
    def update(i):
        #define the current model paramters
        nonlocal W1, b1, W2, b2

        #apply forward propagation
        Z1, A1, Z2, A2 = grad_desc.forward_prop(W1, b1, W2, b2, X)

        #apply back propagation
        dW1, db1, dW2, db2 = grad_desc.back_prop(Z1, A1, Z2, A2, W2, X, Y)

        #update the current model parameters
        W1, b1, W2, b2 = grad_desc.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        #calculate the current models accuracy
        y_pred = grad_desc.predict(A2)
        accuracy = grad_desc.get_accuracy(y_pred, Y)

        #update scatter plot
        ax.set_title(f"Iteration: {i} - Accuracy: {accuracy:.2f}")
        print(f"Iteration: {i} - Accuracy: {accuracy}")

        #draw decision boundaries
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z1, A1, Z2, A2 = grad_desc.forward_prop(W1, b1, W2, b2, pca.inverse_transform(grid).T)
        Z = grad_desc.predict(A2).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha = 0.3, levels = np.unique(Z), cmap = 'Set1')

        #plot the training data PC scores
        scatter = ax.scatter(X_pca[0, :], X_pca[1, :], s = 3, c = Y, cmap = 'Set1')

    #animate the training and save as GIF
    ani = animation.FuncAnimation(fig, update, frames = iterations, repeat = False)
    writer = animation.PillowWriter(fps = 10, bitrate = 1500)
    ani.save(f"{DIR}/images/model_training_animation.gif", writer = writer)

#create the training animation
gradient_descent_with_animation(X_train, y_train, iterations = 250, alpha = 0.2)
