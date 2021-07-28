from csv import reader
from math import exp
import numpy as np
from sklearn import metrics
from random import random
np.random.seed(10)
def _csv(csv):
    rices = list()
    with open(csv, 'r') as file:
        csv_r = reader(file)
        for r in csv_r:
            if not r:continue
            rices.append(r)
    return rices
def convertStrtoFlo(ricedataset, col):
    for r in ricedataset:r[col] = float(r[col].strip())
def convertStrtoInt(ricedataset, col):
    cvalues = [r[col] for r in ricedataset]
    u = set(cvalues)
    look = dict()
    for i, value in enumerate(u):look[value] = i
    for r in ricedataset:r[col] = look[r[col]]
    return look
def _minmax(ricedatas):
    stats = [[min(column), max(column)] for column in zip(*ricedatas)]
    return stats
def normalize(ricedatas, minmax):
    for r in ricedatas:
        for i in range(len(r) - 1):
            r[i] = (r[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
def evaluate_algo(train_s, test_s, algo, *args):
    predct = algo(train_s, test_s, *args)
    actual = [r[-1] for r in test_s]
    accr = metrics.accuracy_score(actual, predct)
    return accr
def active(weight, inpt):
    active = weight[-1]
    for i in range(len(weight) - 1):
        active += weight[i] * inpt[i]
    return active
def trnsfer(active):return 1.0 / (1.0 + exp(-active))
def trnsferderive(outpt):return outpt * (1.0 - outpt)
def frward_propgate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = active(neuron['weights'], inputs)
            neuron['output'] = trnsfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
def bckward_propgate_err(network, expcted):
    for i in reversed(range(len(network))):
        layer = network[i]
        errs = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                err = 0.0
                for neuron in network[i + 1]:
                    err += (neuron['weights'][j] * neuron['delta'])
                errs.append(err)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errs.append(expcted[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errs[j] * trnsferderive(neuron['output'])
def change_weight(netwrk, r, learning_rate):
    for i in range(len(netwrk)):
        inpts = r[:-1]
        if i != 0:
            inpts = [neuron['output'] for neuron in netwrk[i - 1]]
        for neuron in netwrk[i]:
            for j in range(len(inpts)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inpts[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']
def train_net(netwrk, train, learning_rate, number_epch, number_outp):
    errs = []
    for epch in range(number_epch):
        sum_err = 0
        for r in train:
            outp = frward_propgate(netwrk, r)
            expectd = [0 for i in range(number_outp)]
            expectd[r[-1]] = 1
            bckward_propgate_err(netwrk, expectd)
            sum_err += sum([(expectd[i] - outp[i]) ** 2 for i in range(len(expectd))])
            change_weight(netwrk, r, learning_rate)
            errs.append(sum_err)
        print('epoch number=%d, learning rate=%.1f, error=%.1f' % (epch, learning_rate, sum_err))
def start_net(number_inpt, number_h, number_outpt):
    netwrk = list()
    h_layer = [{'weights': [random() for i in range(number_inpt + 1)]} for i in range(number_h)]
    netwrk.append(h_layer)
    o_layer = [{'weights': [random() for i in range(number_hidden + 1)]} for i in range(number_outpt)]
    netwrk.append(o_layer)
    return netwrk
def guess(network, r):
    outpt = frward_propgate(network, r)
    return outpt.index(max(outpt))
def back_prop_multi_percep(train, test, learning_rate, number_epoch, number_hidden):
    n_i = len(train[0]) - 1
    n_o = len(set([row[-1] for row in train]))
    netwrk = start_net(n_i, number_hidden, n_o)
    train_net(netwrk, train, learning_rate, number_epoch, n_o)
    guessings = list()
    for r in test:
        guessing = guess(netwrk, r)
        guessings.append(guessing)
    return (guessings)
csvtrain = 'Ricetrain.csv'
csvtest = 'Ricetest.csv'
test_data = _csv(csvtest)
train_data = _csv(csvtrain)
for i in range(len(train_data[0]) - 1):convertStrtoFlo(train_data, i)
for i in range(len(test_data[0]) - 1):convertStrtoFlo(test_data, i)
convertStrtoInt(train_data, len(train_data[0]) - 1)
minmax = _minmax(train_data)
normalize(train_data, minmax)
convertStrtoInt(test_data, len(test_data[0]) - 1)
minmax = _minmax(test_data)
normalize(test_data, minmax)
learning_rate = 0.3
number_epoch = 10
number_hidden = 10
scores = evaluate_algo(train_data, test_data, back_prop_multi_percep, learning_rate, number_epoch, number_hidden)
print('Accuracy: %s' % scores)
