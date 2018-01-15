'''
    Time-correlated entropy approximation module described in
    Zhao, Kai, et al. "Predicting taxi demand at high spatial resolution: 
    Approaching the limit of predictability." Big Data (Big Data), 
    2016 IEEE International Conference on. IEEE, 2016.
    
    Given the time series its time-correlated entropy S (real value) is evaluated.
'''

import numpy as np

def is_sublist(list_, sublist):
    # Turns string lists into strings and checks if the sublist is in the list
    # input: 
    #   list_ : list of strings [' ']
    #   sublist : list of strings [' ']
    # output:
    #   True if sublist is in list_, False otherwise
    l1 = "".join(map(str, list_))
    l2 = "".join(map(str, sublist))
    if l2 in l1:
        return True
    return False
    
def shortest_subsequence(series, i):
    # Calculates length of the shortest subsequence at time step i
    # that hasn't appeared before
    # input: 
    #   series: time series, list of strings [' ']
    #   i: time step index, integer
    # output:
    #   length of the shortest subsequence 
    sequences = [series[i]]  
    count = 1
    while is_sublist(series[:i], sequences) and i + count <= len(series) - 1:
        sequences = sequences + [series[i+count]]
        count +=1
    return len(sequences)

def RealEntropy(timeseries):
    # Calculates an approximation of the time-correlated entropy.
    # input:
    #   timeseries: sequence of numbers.
    # output:
    #   RealEntropy: approximation of the time-correlated entropy of the time series.
    timeseries = map(str, timeseries)
    substring_length_gen = (shortest_subsequence(timeseries, i) for i in range(1, len(timeseries)))
    shortest_substring_lengths = [1] + map(lambda length: length, substring_length_gen)
    RealEntropy = np.log(len(timeseries)) * len(timeseries) / np.sum(shortest_substring_lengths)
    return RealEntropy
