import numpy as np
import random

class Node:
    
    def __init__(self, weights : list[float], b : float) -> None:
        self.weights = weights
        self.numWeights = len(weights)
        self.b = b

    def sigmoidActualize(self, features : list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        prediction = 1/(1 + np.exp(-z))
        return prediction
    
    def setWeights(self, weights : list[float], b):
        self.weights = weights
        self.b = b
        self.numWeights = len(weights)
    
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.b
    
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
    
    def denseNumpy(self, inputLayer : np.ndarray | list[list[float]]) -> np.ndarray:
        z = np.matmul(inputLayer, self.getWeights().T) + self.getBiases()
        out = self.sigmoid(z)
        return out

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))
    
    def printLayer(self):
        for i in range(self.numNodes):
            print(f"Node {i+1}:", self.nodes[i])

    def getWeights(self):
        return [node.getWeights() for node in self.nodes]
    
    def getBiases(self):
        return [[node.getBias()] for node in self.nodes]

    def getNodes(self):
        return self.nodes
    
class NeuralNetwork:

    def __init__(self, layers : list[Layer]) -> None:
        self.layers = layers
        self.numLayers = len(layers)

    def addLayer(self, layer : Layer, location=0):
        self.layers.insert(location, layer)
        numLayers+=1

    def predict(self, inputLayer : np.ndarray):
        
        outputs = []

        input = inputLayer
        for layer in self.layers:
            output = layer.denseNumpy(input)
            outputs.append(output)

            input = output

        print(outputs)
        return output[-1]
    
    def printLayers(self):
        for i in range(self.numLayers):
            print(f"Layer {i+1}: ") 
            self.layers[i].printLayer()