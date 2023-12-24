import statistics as stat

class LinAlg:

    def __init__(self) -> None:
        print('Linear Algebra class initialized')

    def mean(self, vector):
        return stat.fmean(vector)
    
    def slope(self, X, y):
        meanX = self.mean(X)
        meanY = self.mean(y)
        numerator = 0
        denominator = 0

        for currX, currY in X, y:
            numerator += (currX - meanX) * (currY - meanY)
            denominator += (currX - meanX) ** 2
        
        return numerator/denominator

    def yIntercept(self, X, y, slope):
        return self.mean(y) - slope * self.mean(X) 