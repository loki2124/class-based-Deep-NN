import numpy as np
import matplotlib.pyplot as plt
import h5py
from timeit import default_timer
from functools import wraps

#Timer Decorator to check the time elapsed per function call
def timer(func):
    """Print the runtime of the decorated function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = default_timer()
        func_return_val = func(*args, **kwargs)
        end = default_timer()
        print("{0:<10} {1:<8} {2:^8}".format("module\t", "function\t", "time taken\t"))
        print(
            "{0:<10} {1:<8} {2:^8}".format(func.__module__, func.__name__, end - start)
        )
        return func_return_val
    return wrapper


#Data Loader
def load_data():
    """
    Loads Train & Test dataset from h5 file
    
    Arguments:
    Null 
    
    Returns:
    train_set_x_orig -- Training images
    train_set_y_orig -- Training image labels
    test_set_x_orig  -- Test images 
    test_set_y_orig  -- Test image labels 
    classes          -- Number of classes
    """

    train_dataset = h5py.File('../input/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('../input/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#Sigmoid Function
def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

#Relu function
def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

