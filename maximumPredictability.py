'''
    Maximum predictability evaluation module described in
    Zhao, Kai, et al. "Predicting taxi demand at high spatial resolution: 
    Approaching the limit of predictability." Big Data (Big Data), 
    2016 IEEE International Conference on. IEEE, 2016.
    
    Given the number of unique values N in the time series and its time-correlated entropy S
    the module approximates the theoretical limit of the series predictability Pmax.
'''

import numpy as np

def Function(x, N, S):
    # Calculates the value of the maximum predictability function.
    # inputs:
    #   x (evaluated maximum predictability, Pmax) real number, 
    #   N number of unique values in the time series, 
    #   S time-correlated entropy of the time series.
    # output: 
    #   real value of the function.
    return 1.0*(-x*np.log(x)-(1-x)*np.log(1-x)+(1-x)*np.log(N-1)-S*np.log(2))

def FirstDerivative(x, N):
    # Calculates the value of the first derivative of the maximum predictability function.
    # inputs:
    #   x (evaluated maximum predictability, Pmax) real number, 
    #   N number of unique values in the time series.
    # output:
    #   real value of the first derivative.
    return 1.0*(np.log(1-x)-np.log(x)-np.log(N-1))

def SecondDerivative(x):
    # Calculates the value of the second derivative of the maximum predictability function.
    # inputs:
    #   x (evaluated maximum predictability, Pmax) real number. 
    # output:
    #   real value of the second derivative.
    return 1.0/((x-1)*x)

def CalculateNewApproximation(x, N, S):
    # Calculates a one-step approximation of the value of the maximum predictability function.  
    # inputs:
    #   x (evaluated maximum predictability, Pmax) real number, 
    #   N number of unique values in the time series, 
    #   S time-correlated entropy of the time series.
    # output: 
    #   real value of the function.
    function = Function(x, N, S)
    first_derivative = FirstDerivative(x, N)
    second_derivative = SecondDerivative(x)
    return 1.0*function/(first_derivative-function*second_derivative/(2*first_derivative))

def maximum_predictability(N, S):
    # Evaluates the value of the maximum predictability of the series.
    # inputs:
    #   N number of unique values in time series, integer,
    #   S time-correlated entropy of the time series, real.
    # output:
    #   value of the maximum predictability, Pmax.
    S = round(S, 9)
    if S > round(np.log2(N), 9):
        return "No solutions"
    else:
        if S <= 0.01:
            return 0.999
        else:
            x = 1.0000000001/N
            while abs(Function(x, N, S))>0.00000001:
                x = x - CalculateNewApproximation(x, N, S)
    return round(x, 10)
