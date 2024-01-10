pub mod stochastic {
    use crate::gradient_descent::obj::obj::GradientDescent;

    /// Performs batch gradient descent. Must have an already created
    /// Gradient Descent object with a int of epochs to perform
    pub fn batch(gd: GradientDescent, epochs: i32) {
        let m: i32 = gd.x_train.len() as i32;
        for i in 0..epochs {
            for j in 0..gd.num_predictors {
                let mut sum: f32 = 0.0;
                for (predictors, output) in gd.train_data() {
                    sum += (gd.h(predictors.to_vec()) - output) * predictors[j]
                }
            }
        }
    }
}