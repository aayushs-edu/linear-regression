import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

from gradient_descent import BatchGradientDescent

df = pd.read_csv('data/50_Startups.csv')

# For plotting
xPlot = df['R&D Spend']
zPlot = df['Marketing Spend']
yPlot = df['Profit']


X = df.iloc[:, :-2].values
y = df.iloc[:, 4].values

fig = plt.figure()
ax = plt.axes(projection='3d')


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

ax.scatter3D(xPlot, zPlot, yPlot)
ax.set_title('Gradient Descent')
ax.set_xlabel(f'{df.columns[0]}')
ax.set_ylabel(f'{df.columns[2]}')
ax.set_zlabel('Profit')


bgd = BatchGradientDescent(X_train.tolist(), y_train.tolist(), 3)

bgd.optimizeTheta(1000)

y_preds = [bgd.h(x) for x in X_test]

comparison = np.column_stack((y_test, y_preds))

# print(comparison)

comparison = scaler.fit_transform(comparison)

y_test, y_preds = list(zip(*comparison))

# print(mean_squared_error(y_test, y_preds)) 

x = np.linspace(0, max(xPlot), len(xPlot))
z = np.linspace(0, max(zPlot), len(zPlot))

# plotInput = X_train.tolist() + X_test.tolist()

print(bgd.getyInt())

plotScaled = scaler.fit_transform(list(zip(x, z)))

ax.plot3D(x, z, [bgd.h(x) for x in plotScaled], color ='red')



plt.show()
