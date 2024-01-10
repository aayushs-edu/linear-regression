import random
import numpy as np

class GradientDescent:
    def __init__(self, xTrain, yTrain, numPredictors, learningRate):
        self.thetaVector = [random.uniform(0, 1) for _ in range(numPredictors)]
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.b = random.uniform(0, 1)
        self.learningRate = learningRate
        self.numPredictors = numPredictors


    def batchGradientDescent(self, epochs : int):

        m = len(self.xTrain)

        for i in range(epochs):
            for j in range(self.numPredictors):
                sum = 0
                for (predictors, output) in self.trainData():
                    sum += (self.h(predictors) - output) * predictors[j]
                self.thetaVector[j] -= (1/m) * self.learningRate * sum
            sum = 0
            for (predictors, output) in self.trainData():
                sum += (self.h(predictors) - output)
            self.b -= (1/m) * self.learningRate * sum
            print(f'Epoch {i}: ', self.thetaVector, 'Cost: ', self.cost(self.thetaVector, self.b))
            
    def batchGDSimul(self, epochs : int):

        m = len(self.xTrain)

        for i in range(epochs):
            for (predictors, output) in self.trainData():
                error = self.h(predictors) - output
                for j in range(self.numPredictors):
                    self.thetaVector[j] -= (self.learningRate/m) * error * predictors[j]
                self.b -= (self.learningRate/m) * error
            print(f'Epoch {i}: ', self.thetaVector, 'Cost: ', self.cost(self.thetaVector, self.b))

    def batchGDVectorized(self, epochs : int):

        m = len(self.xTrain)

        for i in range(epochs):
            for (predictors, output) in self.trainData():
                error = self.hVectorized(predictors) - output
                for j in range(self.numPredictors):
                    self.thetaVector[j] -= (self.learningRate/m) * error * predictors[j]
                self.b -= (self.learningRate/m) * error
            print(f'Epoch {i}: ', self.thetaVector, 'Cost: ', self.cost(self.thetaVector, self.b))
    
    def stochasticGDVectorized(self, epochs : int, batchSize):

        m = len(self.xTrain)

        for i in range(epochs):
            for (predictors, output) in self.trainData()[i:i+batchSize]:
                error = self.hVectorized(predictors) - output
                for j in range(self.numPredictors):
                    self.thetaVector[j] -= (self.learningRate/m) * error * predictors[j]
                self.b -= (self.learningRate/m) * error
            print(f'Epoch {i}: ', self.thetaVector, 'Cost: ', self.cost(self.thetaVector, self.b))
    

    def cost(self, thetaVector, b):
        m = len(self.xTrain)
        cost = 0
        for (predictors, output) in self.trainData():
            cost += (self.hGivenParams(predictors, thetaVector, b) - output)**2
        return cost * 1/(2 * m)

    def h(self, X):
        result = 0
        for i, x in enumerate(X):
            result += self.thetaVector[i] * x
        return result + self.b
    
    def hVectorized(self, X):
        return np.dot(self.thetaVector, X) + self.b
    
    def hGivenParams(self, X, thetaVector, b):
        return np.dot(thetaVector, X) + b
    
    def getyInt(self):
        return self.b
    
    def getParams(self):
        return self.thetaVector
    
    def trainData(self):
        return list(zip(self.xTrain, self.yTrain))

