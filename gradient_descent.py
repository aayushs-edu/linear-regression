import random

class BatchGradientDescent:
    def __init__(self, xTrain, yTrain, numPredictors):
        self.thetaVector = [random.uniform(0, 1) for _ in range(numPredictors)]
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.b = random.uniform(0, 1)
        self.learningRate = 0.03
        self.numPredictors = numPredictors

    def optimizeTheta(self):
        for i in range(100):
            for j in range(self.numPredictors):
                sum = 0
                for (predictors, output) in list(zip(self.xTrain, self.yTrain)):
                    sum += (self.h([x for x in predictors]) - output) * predictors[j]
                self.thetaVector[j] -= self.learningRate * sum
            sum = 0
            for (predictors, output) in list(zip(self.xTrain, self.yTrain)):
                sum += (self.h([x for x in predictors]) - output)
            self.b -= self.learningRate * sum

            print(self.thetaVector)
   
    def h(self, X):
        result = 0
        for i, x in enumerate(X):
            result += self.thetaVector[i] * x
        return result + self.b
    
