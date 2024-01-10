// contains the GD struct

pub mod obj {

    use rand::Rng;

    pub struct GradientDescent {
        pub theta_vector: Vec<f32>,
        pub b: f32,
        pub learning_rate: f32,
        pub num_predictors: usize,
        pub x_train: Vec<f32>,
        pub y_train: Vec<f32>
    }

    impl GradientDescent {

        /// Creates a new GradientDescent object
        pub fn new(x_train: Vec<f32>, y_train: Vec<f32>, num_predictors: usize, learning_rate: f32) -> GradientDescent {

            let mut theta_vector: Vec<f32> = Vec::with_capacity(num_predictors);
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            theta_vector.extend((0..num_predictors).map(|_| rng.gen_range(0.0..1.0)));
            
            let b: f32 = rng.gen_range(0.0..1.0);

            GradientDescent {
                theta_vector,
                b,
                learning_rate,
                num_predictors,
                x_train,
                y_train
            }
        }
        
        pub fn h(&self, X: Vec<f32>) -> f32 {
            let mut result: f32 = 0.0;
            for (i, x) in X.iter().enumerate() {
                result += (self.theta_vector[i] * x).clone()
            }
            result + self.b
        }

        pub fn h_vectorized(&self, X: Vec<f32>) -> f32 {
            self.theta_vector.iter().zip(X.iter()).fold(0.0, |acc, (&theta, &x)| acc + theta * x) + self.b
        }
        
        pub fn h_given_params(&self, X: Vec<f32>, theta_vector: Vec<f32>, b: f32) -> f32 {
            theta_vector.iter().zip(X.iter()).fold(0.0, |acc, (&theta, &x)| acc + theta * x) + b
        }

        pub fn get_y(&self) -> f32 {
            self.b.clone()
        }

        pub fn get_params(&self) -> Vec<f32> {
            self.theta_vector.clone()
        }

        pub fn train_data(&self) -> Vec<(Vec<f32>, f32)> {
            self.x_train.iter().cloned().zip(self.y_train.iter().cloned()).collect()
        }

        pub fn cost(&self, theta_vector: Vec<f32>, b: f32) -> f32 {
            let m: f32 = self.x_train.len() as f32;
            let mut cost: f32 = 0.0;
            for (predictors, output) in self.train_data() {
                cost += (self.h_given_params(predictors.to_vec(), theta_vector, b) - output).powf(2.0);
            }
            cost * 1.0 / (2.0 * m)
        }   

    }
}