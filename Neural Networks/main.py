from NeuralNetwork import NeuralNetwork, Layer
import numpy as np


layer1 = Layer(2)
layer1.setAllWeights([[0.5, 0.6], [0.7, 0.8]], [0.07, 0.05])

layer2 = Layer(1)
layer2.setAllWeights([[0.05, 0.2]], [0.03])

nn = NeuralNetwork([layer1, layer2])

X = np.random.random((2, 5))
Y = np.array([[1.0, 0.0, 1.0, 0.0, 0.0]])

A = np.array([[0.6, 0.3, 0.7, 0.6, 0.76]])



assert layer1.getWeights().shape == (2, 2)
print(list(reversed(range(5))))
print(nn.backProp('sigmoid', X, Y))





