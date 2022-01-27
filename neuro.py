import numpy as np
import scipy.special

class neuralNetwork():

    def __init__(self, inputnodes, hiddennodes, outputnodes, learninggrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learninggrate = learninggrate

        self.wih = (np.random.rand(self.hiddennodes, self.inputnodes) - 0.5)
        self.who = (np.random.rand(self.outputnodes, self.hiddennodes) - 0.5)

        self.activation = lambda x: scipy.special.expit(x)

    def train(self, input_lists, target_lists):
        inputs = np.array(input_lists, ndmin=2).T
        targets = np.array(target_lists, ndmin=2).T

        hidden_inp = np.dot(self.wih, inputs)
        hidden_out = self.activation(hidden_inp)
        final_inputs = np.dot(self.who, hidden_out)
        final_out = self.activation(final_inputs)

        output_errors = targets - final_out
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.learninggrate * np.dot((output_errors * final_out * (1.0 - final_out)), np.transpose(hidden_out))
        self.wih += self.learninggrate * np.dot((hidden_errors * hidden_out * (1.0 - hidden_out)), np.transpose(inputs))


    def query(self, input_lists):
        inputs = np.array(input_lists, ndmin=2).T
        hidden_inp = np.dot(self.wih, inputs)
        hidden_out = self.activation(hidden_inp)
        final_inputs = np.dot(self.who, hidden_out)
        final_out = self.activation(final_inputs)
        return final_out
