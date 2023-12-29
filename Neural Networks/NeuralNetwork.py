import numpy as np

class Node:
    
    def __init__(self, weights : list[float], b : float) -> None:
        self.weights = weights
        self.b = b

    def sigmoidActualize(self, features : list[float]) -> float:
        z = np.dot(features, self.weights) + self.b
        prediction = 1/(1 + np.exp(-z))
        return prediction
    
    def setWeights(self, weights : list[float], b):
        self.weights = weights
        self.b = b
    
    def getWeights(self):
        return self.weights, self.b
    
class Layer:

    def __init__(self, numNodes) -> None:
        self.nodes = [Node([0], 0)] * numNodes
        self.numNodes = numNodes

    def setWeights(self, index : int, weights : list[float], bias : float):
        self.nodes[index].setWeights(weights, bias)

    def setAllWeights(self, weights : list[list[float]], biases : list[float]):

        if len(weights) != len(biases): raise Exception('Weights and biases lists must be same length array')

        for i, weight, bias in enumerate(list(zip(weights, biases))):
            self.nodes[i].setWeights(weight, bias)

    def executeLayer(self, inputLayer : list[float]) -> list[float]:

        numfeatures = len(inputLayer)

        aVector = []
        for node in self.nodes:
            if numfeatures != node.numWeights:
                raise Exception('Number of weights does not match number of features')
            aVector.append(node.sigmoidActualize(inputLayer))

        return aVector
    
class NeuralNetwork:

    def __init__(self, layers : list[Layer]) -> None:
        self.layers = layers
        self.numLayers = len(layers)

    def addLayer(self, layer : Layer, location=0):
        self.layers.insert(location, layer)
        numLayers+=1

    def predict(self, inputLayer):
        for i in range(self.numLayers):



    

    

    

        

