import random

class BatchGradientDescent:
    def __init__(self, xTrain, yTrain, numPredictors, learningRate):
        self.thetaVector = [random.uniform(0, 1) for _ in range(numPredictors)]
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.b = random.uniform(0, 1)
        self.learningRate = learningRate
        self.numPredictors = numPredictors


# FIX THIS -- need to update all thetas SIMULTANEOUSLY

    def optimizeTheta(self, epochs):

        m = len(self.xTrain)

        for i in range(epochs):
            for j in range(self.numPredictors):
                sum = 0
                for (predictors, output) in list(zip(self.xTrain, self.yTrain)):
                    sum += (self.h([x for x in predictors]) - output) * predictors[j]
                self.thetaVector[j] -= (1/m) * self.learningRate * sum
            sum = 0
            for (predictors, output) in list(zip(self.xTrain, self.yTrain)):
                sum += (self.h([x for x in predictors]) - output)
            self.b -= (1/m) * self.learningRate * sum

            print(self.thetaVector)
        
        
   
    def h(self, X):
        result = 0
        for i, x in enumerate(X):
            result += self.thetaVector[i] * x
        return result + self.b
    
    def getyInt(self):
        return self.b

    
