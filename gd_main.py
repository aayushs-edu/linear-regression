import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gradient_descent import BatchGradientDescent

df = pd.read_csv('data/50_Startups.csv')

X = df.iloc[:, :-2].values
y = df.iloc[:, 4].values

scaler = StandardScaler()

X = scaler.fit_transform(X)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

bgd = BatchGradientDescent(X_train.tolist(), y_train.tolist(), 3)

bgd.optimizeTheta()

for x in X_test:
    print(bgd.h(x))