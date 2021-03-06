# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.dates as md
from sklearn.neural_network import MLPRegressor
from parametres import input_sizes, layer_sizes

__author__ = 'Anton Korobkov'

# Load the data and save it into appropriate structure, we also need to figure out how much rows we have
observations = np.loadtxt('gold.dlm',  converters={0: md.datestr2num, 1: float}, skiprows=1)


def construct_train(train_length, **kwargs):
    """
    Train and test model with given input
    window and number of neurons in layer
    """
    start_cur_postion = 0
    steps, steplen = observations.size/(2 * train_length), train_length

    if 'hidden_layer' in kwargs:
        network = MLPRegressor(hidden_layer_sizes=kwargs['hidden_layer'])
    else:
        network = MLPRegressor()

    quality = []

    # fit model - configure parameters
    network.fit(observations[start_cur_postion:train_length][:, 1].reshape(1, train_length),
                observations[:, 1][start_cur_postion:train_length].reshape(1, train_length))

    parts = []

    # calculate predicted values
    # for each step add all predicted values to a list
    # TODO: add some parallelism here
    for i in xrange(0, steps):
        parts.append(network.predict(observations[start_cur_postion:train_length][:, 1]))
        start_cur_postion += steplen
        train_length += steplen

    # estimate model quality using 
    result = np.array(parts).flatten().tolist()
    for valnum, value in enumerate(result):
        quality.append((value - observations[valnum][1])**2)

    return sum(quality)/len(quality)


def main():
    result_list = open('analysis_results.txt', 'w')

    for size in input_sizes:
        for layer in layer_sizes:
            result_list.write(' '.join(['Input window:', str(size), 'Neurons:', str(layer), 'Resulting error:',
                                        str(construct_train(size, hidden_layer=layer)), '\n']))

if __name__ == "__main__":
    main()
