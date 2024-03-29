import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    #########################################
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis = np.sum(X[i,:] * theta[:]) 
    ########################################/
    
    return hypothesis
