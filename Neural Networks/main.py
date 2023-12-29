from NeuralNetwork import NeuralNetwork, Node, Layer
import numpy as np


layer1 = Layer(2)
# layer1.setWeights(0, [0.5, 0.6], 1.0)
# layer1.setWeights(1, [0.7, 0.8], 2.0)
layer1.setAllWeights([[0.5, 0.6], [0.7, 0.8]], [1, 2])
layer1.printLayer()

layer2 = Layer(3)
layer2.setAllWeights([[0.7, 0.23], [0.1, 0.4], [0.3, 1.5]], [4, 10, 8])

nn = NeuralNetwork([layer1, layer2])
nn.printLayers()