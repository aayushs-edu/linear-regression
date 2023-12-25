import statistics as stat

class LinAlg:

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def meanX(self):
        return stat.fmean(self.X)
    
    def meanY(self):
        return stat.fmean(self.y)
    
    def slope(self) -> float:
        meanX = self.meanX()
        meanY = self.meanY()
        numerator = 0
        denominator = 0

        for (currX, currY) in zip(self.X, self.y):
            numerator += (currX - meanX) * (currY - meanY)
            denominator += (currX - meanX) ** 2
        
        return numerator/denominator

    def yIntercept(self, slope) -> float:
        return self.meanY() - slope * self.meanX()