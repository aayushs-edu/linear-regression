import numpy as np
import random
import math

class Node:
    
    def __init__(self, weights : list[float], b : float) -> None:
        self.weights = weights
        self.b = b

    # range 0 - 1
    def sigmoidActualize(self, features : list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        prediction = 1/(1 + np.exp(-z))
        return prediction
    
    # range 0 - infinity
    def reluActualize(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        prediction = max(0, z)
        return prediction

    # range -1 - 1
    def tanhActivation(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        a = np.exp(z)
        b = np.exp(-z)
        prediction = (a - b) / (a + b)
        return prediction

    # enables back propogation for ReLu
    # enables negative signed inputs, which means
    # that the gradient on the left side of the activation graph
    # is non zero, enabling back propogation
    def leakyReLuActivation(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b 
        prediction = max(0.1 * z, z)
        return prediction

    # parametric Relu can be used when leaky Relu doesnt solve the
    # zero gradient problem for Relu activation
    # creates problems because the solution is to use a slope value
    # for negative inputs, but there can be difficulty finding the 
    # correct slope value
    def parametricReluActivation(self, features: list[float], a) -> float:
        z = np.dot(features, self.weights) + self.b
        prediction = max(a * z, z)
        return prediction
    
    # uses log curve to define negative inputs 
    # a helps define the log curve
    def eluActivation(self, features: list[float], a) -> float:
        z = np.dot(features, self.weights) + self.b
        return z if z >= 0 else a * (np.exp(z) - 1)

    # useful for multi-class classification problems
    def softMaxActivation(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        max_x = np.amax(features, 1).reshape(features.shape[0],1) # Get the row-wise maximum
        e_x = np.exp(z - max_x ) # For stability
        return e_x / e_x.sum(axis=1, keepdims=True)

    # consistently outperforms or performs at the same level as Relu activation
    # is literally just z * sigmoidActualize(z)
    def swish(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b 
        sigmoid = 1 / (1 + np.exp(-z))
        return z * sigmoid

    # GELU implementation
    def geluActivation(self, features: list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        coefficient = math.sqrt(2 / math.pi)
        return 0.5 * z * (1 + np.tanh(coefficient * (z + 0.044715 * z**3)))

    def setWeights(self, weights : list[float], b):
        self.weights = weights
        self.b = b
    
    def getWeights(self):
        return self.weights, self.b
    
    def __str__(self) -> str:
        return f"Weights: {self.weights}, bias: {self.b}"
    
class Layer:

    def __init__(self, numNodes) -> None:
        self.nodes = [Node([random.random()], random.random()) for _ in range(numNodes)]
        self.numNodes = numNodes

    def setWeights(self, index : int, weights : list[float], bias : float):
        self.nodes[index].setWeights(weights, bias)

    def setAllWeights(self, weights : list[list[float]], biases : list[float]):
        if len(weights) != len(biases): raise Exception('Weights and biases lists must be same length array')

        print(list(enumerate((zip(weights, biases)))))

        for i in range(self.numNodes):
            self.nodes[i].setWeights(weights[i], biases[i])
            

    def executeLayer(self, inputLayer : list[float]) -> list[float]:

        numfeatures = len(inputLayer)

        aVector = []
        for node in self.nodes:
            if numfeatures != node.numWeights:
                raise Exception('Number of weights does not match number of features')
            aVector.append(node.sigmoidActualize(inputLayer))

        return aVector
    
    def printLayer(self):
        for i in range(self.numNodes):
            print(f"Node {i+1}:", self.nodes[i])

    def getNodes(self):
        return self.nodes
    
class NeuralNetwork:

    def __init__(self, layers : list[Layer]) -> None:
        self.layers = layers
        self.numLayers = len(layers)

    def addLayer(self, layer : Layer, location=0):
        self.layers.insert(location, layer)
        numLayers+=1

    def predict(self, inputLayer):
        
        outputs = []

        input = inputLayer
        for layer in self.layers:
            output = [node.sigmoidActualize(input) for node in layer.getNodes()]
            outputs.append(output)

            input = output

        print(outputs)
        return output[0]
    
    def printLayers(self):
        for i in range(self.numLayers):
            print(f"Layer {i+1}: ") 
            self.layers[i].printLayer()