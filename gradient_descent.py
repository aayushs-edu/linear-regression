class BatchGradientDescent:
    def __init__(self, num_predictors : int, learning_rate : float, num_dataset_elements : int, X_train, y_train):
        ''' 
        X_train must be a list of samples, where each sample is a list of features
        y_train must be a list of corresponding labels
        '''
        self.num_predictors = num_predictors
        self.parameters : [list[int]] = [0 for _ in range(self.num_predictors)]
        self.learning_rate = learning_rate
        self.num_dataset_elements = num_dataset_elements
        self.X_train = X_train
        self.y_train = y_train
    def cost(self) -> float:
        r_val : float = 0.0
        for i in range(self.num_dataset_elements):
            r_val += (self.hypothesis(self.X_train[i]) - self.y_train[i]) ** 2
        return r_val / (2 * self.num_dataset_elements)
    def hypothesis(self, x : list) -> float:
        r_val : float = 0.0
        for i in range(self.num_predictors):
            r_val += self.parameters[i] * x[i]
        return r_val
    def step(self) -> None:
        gradients = [0.0 for _ in range(self.num_predictors)]
        for i in range(self.num_dataset_elements):
            for j in range(self.num_predictors):
                gradients[j] += (self.hypothesis(self.X_train[i]) - self.y_train[i] * self.X_train[i][j])
        for j in range(self.num_predictors):
            self.parameters[j] = self.parameters[j] - (self.learning_rate * gradients[j])