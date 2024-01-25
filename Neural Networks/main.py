from NeuralNetwork import NeuralNetwork, Layer
import numpy as np


layer1 = Layer(2)
layer1.setAllWeights([[0.5, 0.6], [0.7, 0.8]], [[0.07], [0.05]])

layer2 = Layer(1)
layer2.setAllWeights([[0.05, 0.2]], [[0.03]])

nn = NeuralNetwork([layer1, layer2], 0.001)

#X = np.random.random((2, 100))
#Y = np.array(np.random.rand(1, 100) >= 0.5).astype(float)
X = np.array(list(zip(np.linspace(0, 100, num=100), np.linspace(0, 100, num=100)))).T
print("X: ", X.shape)
Y = np.array([[i > 50 for i in range(100)]])
print(Y)

print('Layer 1 weights BEFORE: ', layer1.getWeights(), 'biases BEFORE: ', layer1.getBiases())
print('Layer 2 weights BEFORE: ', layer2.getWeights(), 'biases BEFORE: ', layer2.getBiases())

nn.gradientDescent('sigmoid', X, Y, 100)

print('Layer 1 weights AFTER: ', layer1.getWeights())
print('Layer 2 weights AFTER: ', layer2.getWeights())

import matplotlib.pyplot as plt

# Plotting X
plt.scatter(X[0], X[1], c=Y[0], cmap='viridis')
plt.title('Input Data X')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Plotting Y
plt.plot(Y[0], marker='o')
plt.title('Output Data Y')
plt.xlabel('Index')
plt.ylabel('Y')
plt.show()



