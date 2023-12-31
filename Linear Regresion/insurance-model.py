import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

from gradient_descent import GradientDescent

df = pd.read_csv('data/insurance.csv')

# For plotting
xPlot = df['bmi']
zPlot = df['age']
yPlot = df['charges']


X = df[['age', 'bmi', 'children']].iloc[:, :].values
y = df.iloc[:, -1].values

fig = plt.figure()
ax = plt.axes(projection='3d')

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

print(X_train)

# ax.scatter3D(xPlot, zPlot, yPlot)
# ax.set_title('Gradient Descent')
# ax.set_xlabel('BMI')
# ax.set_ylabel('Age')
# ax.set_zlabel('Insurance')


bgd = GradientDescent(X_train.tolist(), y_train.tolist(), 3, 0.1)

bgd.batchGradientDescent(50)

y_preds = [bgd.h(x) for x in X_test]

comparison = np.column_stack((y_test, y_preds))

print(comparison)

comparison = scaler.fit_transform(comparison)

y_test, y_preds = list(zip(*comparison))

print('Mean squared Error: ', mean_squared_error(y_test, y_preds)) 

# x = np.linspace(min(xPlot), max(xPlot), len(xPlot))
# z = np.linspace(min(zPlot), max(zPlot), len(zPlot))

# plotInput = X_train.tolist() + X_test.tolist()

# print(bgd.getyInt())

# plotScaled = scaler.fit_transform(list(zip(x, z)))

# ax.plot3D(x, z, [bgd.h(x) for x in plotScaled], color ='red')



# plt.show()
