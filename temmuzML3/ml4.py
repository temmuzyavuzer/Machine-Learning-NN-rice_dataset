# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:07:18 2021

@author: sevda
"""

# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import numpy as np

np.random.seed(10)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from random import seed
from random import random
import math
import seaborn as sns
from random import randint
import time


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(X, y, train_set, test_set, algorithm, *args):
    predicted = algorithm(X, y, train_set, test_set, *args)
    actual = [row[-1] for row in test_set]
    accuracy = metrics.accuracy_score(actual, predicted)

    cm = confusion_matrix(actual, predicted)

    plt.figure(figsize=(7, 7))
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%')
    plt.show()

    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.show()

    return accuracy


# Calculate neuron activation for an input
def activate(weights, inputs):
    # print(inputs)
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    # print(activation)
    return 1.0 / (1.0 + np.exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(X, y, network, train, l_rate, n_epoch, n_outputs, batch_size):
    m = len(y)
    n_batches = int(m / batch_size)
    errors = []
    for epoch in range(n_epoch):
        np.random.seed(epoch)
        indices = np.random.permutation(m)
        X_mini = X[indices]
        y_mini = y[indices]
        sum_error = 0
        for i in range(0, m, batch_size):
            X_mini_i = X_mini[i:i + batch_size]
            y_mini_i = y_mini[i:i + batch_size]
            outputs = forward_propagate(network, X_mini_i[0])
            expected = [0 for i in range(n_outputs)]
            expected[int(y_mini_i[0][0])] = 1
            backward_propagate_error(network, expected)
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            update_weights(network, X_mini_i[0], l_rate)
            errors.append(sum_error)
        print('epoch number=%d, learning rate=%.1f, error=%.1f' % (epoch, l_rate, sum_error))


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Make a prediction with a network
def predict(network, X, y, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(X, y, train, test, l_rate, n_epoch, n_hidden, batch_size):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))

    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(X, y, network, train, l_rate, n_epoch, n_outputs, batch_size)
    predictions = list()
    for row in test:
        prediction = predict(network, X, y, row)
        # print(prediction)
        predictions.append(prediction)
    return (predictions)


filename1 = 'Ricetrain.csv'
train_set = load_csv(filename1)

for i in range(len(train_set[0]) - 1):
    str_column_to_float(train_set, i)
# convert class column to integers
str_column_to_int(train_set, len(train_set[0]) - 1)
# normalize input variables
minmax = dataset_minmax(train_set)
normalize_dataset(train_set, minmax)

arr = np.array(train_set)

X = arr[:, [0, 1, 2, 3, 4, 5, 6]]
y = arr[:, -1].reshape((-1, 1))

filename2 = 'Ricetest.csv'
test_set = load_csv(filename2)

for i in range(len(test_set[0]) - 1):
    str_column_to_float(test_set, i)
# convert class column to integers
str_column_to_int(test_set, len(test_set[0]) - 1)
# normalize input variables
minmax = dataset_minmax(test_set)
normalize_dataset(test_set, minmax)
# evaluate algorithm

start1 = time.time()
l_rate = 0.3
n_epoch = 10
n_hidden = 10
batch_size = 30
scores = evaluate_algorithm(X, y, train_set, test_set, back_propagation, l_rate, n_epoch, n_hidden, batch_size)
print('Accuracy: %s' % scores, batch_size)


'''
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
start2 = time.time()
l_rate = 0.3
n_epoch = 20
n_hidden = 5
batch_size = 8
scores = evaluate_algorithm(X,y,train_set,test_set, back_propagation, l_rate, n_epoch, n_hidden, batch_size)
print('Accuracy: %s' % scores,"for batchsize -> ",batch_size)
end2 = time.time()
print("time spent : ",end2 - start2,"for batchsize -> ",batch_size)


print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
start3 = time.time()
l_rate = 0.3
n_epoch = 20
n_hidden = 5
batch_size = 32
scores = evaluate_algorithm(X,y,train_set,test_set, back_propagation, l_rate, n_epoch, n_hidden, batch_size)
print('Accuracy: %s' % scores,"for batchsize -> ",batch_size)
end3 = time.time()
print("time spent : ",end3 - start3,"for batchsize -> ",batch_size)



print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
start4 = time.time()
l_rate = 0.3
n_epoch = 20
n_hidden = 5
batch_size = 64
scores = evaluate_algorithm(X,y,train_set,test_set, back_propagation, l_rate, n_epoch, n_hidden, batch_size)
print('Accuracy: %s' % scores,"for batchsize -> ",batch_size)
end4 = time.time()
print("time spent : ",end4 - start4,"for batchsize -> ",batch_size)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
start5 = time.time()
l_rate = 0.3
n_epoch = 20
n_hidden = 5
batch_size = 128
scores = evaluate_algorithm(X,y,train_set,test_set, back_propagation, l_rate, n_epoch, n_hidden, batch_size)
print('Accuracy: %s' % scores,"for batchsize -> ",batch_size)
end5 = time.time()
print("time spent : ",end5 - start5,"for batchsize -> ",batch_size)

'''







