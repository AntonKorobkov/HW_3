# -*- coding: utf-8 -*-

from pybrain.structure import FeedForwardNetwork, LinearLayer, TanhLayer, FullConnection
from pybrain.datasets.supervised import SupervisedDataSet
import numpy as np
import matplotlib.dates as md

__author__ = 'Anton Korobkov'

# 1. Given the time series (gold.dlm) of prices for gold.
# 2. Construct the neural network for forecasting
#     2.1. Define the length of "input window" (the number of input values for neural network)
#     2.2. Try different numbers of neurons in hidden layer
#     2.3. Perform the forecasting
#            2.3.1. Define the efficient forecasting horyzon (number of forecasted values with an appropriate error)
# 3. Make conclusions on the results of forecasting using different lengths of "input window" and numbers of neurons in hidden layer
#
# Deadline: the end of 3rd module


# Load the data and save it into appropriate structure, we also need to figure out how much rows we have
observations = np.loadtxt('gold.dlm',  converters={0: md.datestr2num, 1: float}, skiprows=1)
train_length = observations.shape[0]/2

# Create 'train' and 'test' sets
train, test = observations[0:train_length], observations[train_length:]

ds = SupervisedDataSet(2, 2)

for rownum in range(0, train_length):
    ds.addSample(train[rownum][0], train[rownum][1])

n = FeedForwardNetwork()
inLayer = LinearLayer(2)
hiddenLayer = TanhLayer(25)
outLayer = LinearLayer(1)
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()

