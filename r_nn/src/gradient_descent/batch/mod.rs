pub mod batch {

    use crate::gradient_descent::obj::obj::GradientDescent;

    /// Performs batch gradient descent. Must have an already created
    /// Gradient Descent object with a int of epochs to perform
    pub fn batch(mut gd: GradientDescent, epochs: i32) {
        let m: i32 = gd.x_train.len() as i32;
        for i in 0..epochs {
            for j in 0..gd.num_predictors {
                let mut sum: f32 = 0.0;
                for (predictors, output) in gd.train_data() {
                    sum += (gd.h(predictors.to_vec()) - output) * predictors[j]
                }
                gd.theta_vector[j] -= (1.0 / m as f32) * gd.learning_rate * sum
            }
            let mut sum: f32 = 0.0;
            for (predictor, output) in gd.train_data() {
                sum += (gd.h(predictor) - output)
            }
            gd.b -= (1.0 / m as f32) * gd.learning_rate * sum;
            println!("Epoch {}, Theta: {:#?}, Cost: {}, ", i, gd.theta_vector, gd.cost(gd.theta_vector.clone(), gd.b))
        }
    }

    pub fn batch_vectorized(mut gd: GradientDescent, epochs: i32) {
        let m: i32 = gd.x_train.len() as i32;
        for i in 0..epochs {
            for (predictors, output) in gd.train_data() {
                let error = gd.h_vectorized(predictors.clone()) - output;
                for j in 0..(gd.num_predictors) {
                    gd.theta_vector[j] -= (gd.learning_rate / m as f32) * error * predictors[j]
                }
                gd.b -= (gd.learning_rate / m as f32) * error
            }
            println!("Epoch: {}, ThetaVector: {:#?}, Cost: {}", i, gd.theta_vector, gd.cost(gd.theta_vector.clone(), gd.b))
        }
    }
}