'''
    Time-correlated entropy approximation module described in
    Zhao, Kai, et al. "Predicting taxi demand at high spatial resolution: 
    Approaching the limit of predictability." Big Data (Big Data), 
    2016 IEEE International Conference on. IEEE, 2016.
    
    Given the time series its time-correlated entropy S (real value) is evaluated.
'''

import math
import numpy as np

def containsSublist(list_, sublist):
    # Tests if the subsequence is present in the sequence.
    # inputs:
    #   list_: sequence of numbers,
    #   sublist: subsequence of numbers.
    # output:
    #   boolean value, True if the subsequence is in the sequence, False otherwise.
    for i in xrange(0, len(list_)-len(sublist)+1):
        if list_[i: i+len(sublist)] == sublist:
            return True
    return False

def RealEntropy(timeseries):
    # Calculates an approximation of the time-correlated entropy.
    # input:
    #   timeseries: sequence of numbers.
    # output:
    #   RealEntropy: approximation of the time-correlated entropy of the time series.
    Shortest_Substring_Length = [1]
    for i in range(1, len(timeseries)):
        sequences = [timeseries[i]]  
        count = 1
        while containsSublist(timeseries[:i], sequences) and i + count <= len(timeseries) - 1:
            sequences.append(timeseries[i+count])
            count +=1
        Shortest_Substring_Length.append(len(sequences))
    RealEntropy = math.log(len(timeseries)) * len(timeseries) / sum(Shortest_Substring_Length)
    return RealEntropy

