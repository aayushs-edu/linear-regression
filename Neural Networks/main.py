from NeuralNetwork import NeuralNetwork, Layer
import numpy as np


layer1 = Layer(2)
layer1.setWeights(0, [0.5, 0.6], 1.0)
layer1.setWeights(1, [0.7, 0.8], 2.0)
layer1.setAllWeights([[0.5, 0.6], [0.7, 0.8]], [0.07, 0.05])

layer2 = Layer(3)
layer2.setAllWeights([[0.7, 0.23], [0.1, 0.4], [0.3, 1.5]], [0.2, 0.5, 0.01])

layer3 = Layer(1)
layer3.setAllWeights([[0.05, 0.2, 0.5]], [0.03])

nn = NeuralNetwork([layer1, layer2, layer3])
# nn.printLayers()

# prediction = nn.predict([0, 0])

cost = nn.binaryCost(np.array([0.8, 0.1, 0.5]), np.array([1, 0, 1]))

print(cost)