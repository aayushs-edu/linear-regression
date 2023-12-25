from gradient_descent import BatchGradientDescent
import random
from sklearn.metrics import mean_squared_error, r2_score
random.seed(42)
num_samples = 100
num_features = 3
true_params = [2.0, -1.5, 3.0]

X_train = [[random.uniform(0, 1) for _ in range(num_features)] for _ in range(num_samples)]
y_train = [true_params[0] * x[0] + true_params[1] * x[1] + true_params[2] * x[2] + random.normalvariate(0, 0.5) for x in X_train]
learning_rate = 0.01
num_predictors = len(X_train[0])
num_dataset_elements = len(X_train)
bgd = BatchGradientDescent(
    num_predictors=num_predictors,
    learning_rate=learning_rate,
    num_dataset_elements=num_dataset_elements,
    X_train=X_train,
    y_train=y_train
)
iterations = 1000
for step in range(iterations):
    bgd.step()

optimized_params = bgd.parameters 
def optimized_hypothesis(x):
    r_val = 0.0
    for i in range(num_predictors):
        r_val += optimized_params[i] * x[i]
    return r_val 

new_sample = [0.7, 0.5, 0.3]
prediction = optimized_hypothesis(new_sample)
print(f'Prediction: {prediction}')
y_pred_train = [optimized_hypothesis(x) for x in X_train]
mse = mean_squared_error(y_train, y_pred_train)
print(f'Mean squared error: {mse}')
r2 = r2_score(y_train, y_pred_train)
print(f'R-squared on training data: {r2}')