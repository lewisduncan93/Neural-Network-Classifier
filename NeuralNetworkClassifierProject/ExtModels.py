#ExtModles
#Holds extenstion functionality for the main classifer
#Neural Network

#NeuralNetworkClassifierProject

#Project Imports
#Import Python numpy for Neural Networks
import numpy
#Import Python random
import random

#Import functions from other Python Classes
#Import learning variables from Configuration class
from Configuration import LEARNING_RATE
#Import Formulas for using sigmoid
from Formulas import sig, inv_sig, inv_err

curr_node_id = 0

#Class Layer Function
class LayerFunction:
    #Initialize 
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    #Evaluate neuron ouput
    def eval(self):
        for x in range(self.num_nodes):
            self.layer_net[x] = numpy.dot(self.input_vals, numpy.transpose(self.weight[x])) + self.bias
            self.layer_out[x] = sig(self.layer_net[x])

    #Back Propagation learning
    def backprop(self, other):
        for x in range(len(self.weight)):
            for y in range(len(self.weight[x])):
                if self.layer_num == 1:
                    self.weight[x][y] = self.weight[x][y] - (LEARNING_RATE * other.weight_delta[0][x] * self.input_vals[y] * other.weight[0][x] * inv_sig(self.layer_out[x]))
                elif self.layer_num == 2:
                    self.weight_delta[x][y] = inv_sig(self.layer_out[x]) * inv_err(self.layer_out[x], other)
                    self.weight[x][y] = self.weight[x][y] - (LEARNING_RATE * self.weight_delta[x][y] * self.input_vals[y])

#Class file grabber
class cfile(file):
    #Define Initialize
    def __init__(self, name, mode = 'r'):
        self = file.__init__(self, name, mode)

    #Define w
    def w(self, string):
        self.writelines(str(string) + '\n')
        return None