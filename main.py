import neuro

import numpy
import matplotlib.pyplot
#%matplotlib inline

"""
inp = 3
hid = 3
outp = 3
lrn = 0.3
"""

#n = neuro.neuralNetwork(inp,hid,outp,lrn)

#print(n.query([1.0, 0.5, -1.5]))
#data_file = open("mnist/tech2.csv", 'r')
#data_list = data_file.readlines()
#data_file.close()
#print(data_list[0])

#all_values = data_list[0].split(",")
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
#matplotlib.pyplot.show()

#scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

inp = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuro.neuralNetwork(inp, hidden_nodes, output_nodes, learning_rate)

training_file = open("mnist/tech2.csv", 'r')
training_list = training_file.readlines()
training_file.close()

for record in training_list:
    all_values = record.split(",")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

test_file = open("mnist/mnist_test.csv")
test_list = test_file.readlines()
test_file.close()


all_values = test_list[0].split(",")
print(all_values[0])

result = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
print(result)

image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()
