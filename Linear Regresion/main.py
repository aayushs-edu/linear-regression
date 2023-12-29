import pandas as pd
import matplotlib.pyplot as plt
from linalg import LinAlg

data = pd.read_csv('data/tvmarketing.csv')

X = data['TV']
y = data['Sales']

print(X)

xData = [float(i) for i in X]
yData = [float(i) for i in y]

la = LinAlg(xData, yData)

plt.scatter(xData, yData)


m = la.slope()
b = la.yIntercept(m)


plt.plot(xData, [(x * m + b) for x in xData], color = 'red')

plt.show()

