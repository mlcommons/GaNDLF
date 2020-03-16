import numpy as np
import math
def cyclical_lr(stepsize, min_lr, max_lr):
    #Scaler : we can adapt this if we do not want the triangular LR
    scaler = lambda x:1
    #Lambda function to calculate the LR
    lr_lambda = lambda it: max_lr - (max_lr - min_lr)*relative(it,stepsize)
    #Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1+it/(2*stepsize))
        x = abs(it/stepsize - 2*cycle + 1)
        return max(0,(1-x))*scaler(cycle)
    return lr_lambda
