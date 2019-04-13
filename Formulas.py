#Formulas class
#Holds details of all mathematical functions
#Neural Network

#NeuralNetworkClassifierProject

#Import Python Maths classes
import math

#Sigmoid Function, for threshold inside Neural Network
def sig(x):
    return float(1) / float(1 + math.exp(-x))

#Inverted Sigmoid Function for testing
def inv_sig(x):
    return sig(x) * (1 - sig(x))

#Return Error function
def err(o, t):
    return 0.5 * ((t - o) ** 2)

#Inverted Error function for testing
def inv_err(o, t):
    return (o - t)