class BatchGradientDescent:
    def __init__(self, num_predictors : int, learning_rate : int, num_dataset_elements : int):
        self.num_predictors = num_predictors
        self.predictors : [list[int]] = [0 for _ in range(self.num_predictors)]
        self.learning_rate = learning_rate
        self.num_dataseta_elements = num_dataset_elements
    def hypothesis(self, X : list[int]) -> int:
        r_val : int = None  
        
    def step(self) -> None:
        for predictor in self.predictors:
            p = predictor
            predictor = p - (self.learning_rate)