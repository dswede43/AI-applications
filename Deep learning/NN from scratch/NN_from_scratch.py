#!/usr/bin/python3

#Neural Network (NN) from scratch
#---
#Script to build a NN from scratch to classify MNIST images.


#Table of contents:
#---
#1. Define training and testing datasets
#2. Build the NN
#3. Model testing


#load libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import gradient_descent_functions as grad_desc

#define global variables
DIR = "/mnt/c/Users/dstra/Nextcloud/Documents/Computers/Code/Github repos/Deep learning/NN from scratch" #working directory

#load data
train_data = pd.read_csv(f"{DIR}/data/mnist_train.csv")
test_data = pd.read_csv(f"{DIR}/data/mnist_test.csv")


#1. Define training and testing datasets
#---
#define feature and target variables
m, n = train_data.shape

train_data = np.array(train_data).T
y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train / 255.0 #normalize data

test_data = np.array(test_data).T
y_test = test_data[0]
X_test = test_data[1:n]
X_test = X_test / 255.0 #normalize data


#2. Build the NN
#---
#estimate model parameters (model training)
W1, b1, W2, b2, train_results = grad_desc.gradient_descent(X_train, y_train, 1000, 0.1)

#format the training results
train_df = pd.DataFrame(train_results).T.reset_index()
train_df.columns = ['iteration','loss','accuracy']

#visualize the training results
plt.plot(train_df['iteration'], grad_desc.sigmoid(train_df['loss']), label = 'loss')
plt.plot(train_df['iteration'], train_df['accuracy'], label = 'accuracy')
plt.legend()
plt.xlabel('Training iteration')
plt.ylabel('')
plt.title('NN training results')
plt.savefig(f"{DIR}/images/training_results.pdf", format = 'pdf')
plt.close()


#3. Model testing
#---
#get testing predictions
y_pred = grad_desc.make_prediction(W1, b1, W2, b2, X_test)

#get testing accuracy
test_accuracy = grad_desc.get_accuracy(y_pred, y_test)
print(f"Testing accuracy: {test_accuracy * 100}%")

#create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

#plot the confusion matrix
plt.figure(figsize = (8,8))
sns.heatmap(cm, annot = True, fmt = '.0f', cmap = 'Blues', cbar = False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title(f"Confusion matrix (NN testing accuracy: {test_accuracy * 100}%)")
plt.savefig(f"{DIR}/images/confusion_matrix.pdf", format = 'pdf')
