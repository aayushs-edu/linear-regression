import numpy as np
import random
import math

class Node:
    
    def __init__(self, weights : list[float], b : float) -> None:
        self.weights = weights
        self.numWeights = len(weights)
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
        self.numWeights = 0

    def setWeights(self, index : int, weights : list[float], bias : float):
        self.nodes[index].setWeights(weights, bias)

    def setAllWeights(self, weights : list[list[float]], biases : list[float]):
        if len(weights) != len(biases): raise Exception('Weights and biases lists must be same length array')
        self.numWeights = len(weights[0])
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
        Z = np.matmul(self.getWeights(), inputLayer) + self.getBiases()
        out = self.sigmoid(Z)
        return out
    
    def sigmoidZ(self, inputLayer : np.ndarray) -> np.ndarray:
        print(self.getWeights().shape)
        print(inputLayer.shape)
        return np.matmul(self.getWeights(), inputLayer) + self.getBiases()
    
    def sigmoidA(self, Z : np.ndarray) -> np.ndarray:
        return self.sigmoid(Z)

    def sigmoid(self, input : np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-input))
    
    def printLayer(self):
        for i in range(self.numNodes):
            print(f"Node {i+1}:", self.nodes[i])

    def getWeights(self) -> np.ndarray:
        return np.array([node.getWeights() for node in self.nodes])
    
    def getBiases(self):
        return np.array([[node.getBias()] for node in self.nodes])

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
    
    def MSE(self, yHats : np.ndarray, yActuals : np.ndarray):
        _sum = np.square(yHats - yActuals)
        return _sum * 1/(len(yActuals) * 2)
    
    def sigmoidCost(self, yHats : np.ndarray, yActuals : np.ndarray):
        _sum = yActuals * np.log2(yHats) + (1 - yActuals) * np.log2(1 - yHats)
        return -1/len(yActuals) * _sum.sum()
        
    def multiclassCost(self, yHats: np.ndarray, yActuals : np.ndarray):
        _sum = np.matmul(yActuals, yHats.T).sum()
        return -1/len(yActuals) * _sum
    
    def backProp(self, costFun : str, X : np.ndarray, Y : np.ndarray):

        L = self.numLayers
        m = X.shape[1]
        A : list[np.ndarray] = [X]
        Z : list[np.ndarray] = [0]


        match costFun:
            case 'MSE':
                pass

            case 'sigmoid':
                
                # Forward Prop
                for i, layer in enumerate(self.layers):
                    if i == 0:
                        assert A[i].shape == (self.layers[i].numWeights, m)
                    else:
                        assert A[i].shape == (self.layers[i-1].numNodes, m)
                    Z.append(layer.sigmoidZ(A[i]))
                    A.append(layer.sigmoidA(Z[i+1]))
                    print(f"Forward Pass layer {i}")
                # Backward Prop
                dZ = []
                dW = []
                db = []
                for l in list(reversed(range(L+1))):
                    dZcurr = None
                    if l == L:
                        dZcurr = np.array(A[l] - Y)
                        dZ.append(dZcurr)
                    else:
                        dA = np.matmul(self.layers[l+1].getWeights().T, dZ[l-1])
                        gPrime = np.multiply(A[l], 1 - A[l])
                        dZcurr = np.multiply(dA, gPrime)
                        dZ.append(dZcurr)
                    dW.append((1/m) * np.matmul(dZcurr, A[l-1].T))
                    db.append((1/m) * np.sum(dZcurr, axis=1, keepdims=True))
                    print(f"Backward Pass layer {l}")
                
                return dW, db

            case 'multiclass':
                pass
    
    def printLayers(self):
        for i in range(self.numLayers):
            print(f"Layer {i+1}: ") 
            self.layers[i].printLayer()

