import csv
import matplotlib.pyplot as plt
from linalg import LinAlg

file = open('data/car data.csv')

data = csv.reader(file)

la = LinAlg()

columns = []
headers = []

for i, row in enumerate(data):
    for index, element in enumerate(row):
        if i == 0: 
            columns.append([])
            headers.append(element)
        else: columns[index].append(element)

dataframe = list(zip(headers, columns))

X = dataframe[4]
y = dataframe[3]

plt.scatter(X[1], y[1])

plt.show()
