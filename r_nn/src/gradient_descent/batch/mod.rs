pub mod batch {
    
    use crate::gradient_descent::obj::obj::GradientDescent;

    pub fn stochastic_vect(mut gd: GradientDescent, epochs: i32) {
        let m: i32 = gd.x_train.len() as i32;

        for i in 0..epochs {
            for (predictor, output) in gd.train_data() {
                let error = gd.h(predictor.clone()) - output;
                for j in 0..gd.num_predictors {
                    gd.theta_vector[j] -= (gd.learning_rate / m as f32) * error * predictor[j]
                }
                gd.b -= (gd.learning_rate / m as f32) * error
            }
            println!("Epoch: {}, Theta Vector: {:#?}, Cost: {}", i, gd.theta_vector, gd.cost(gd.theta_vector.clone(), gd.b))
        }
    }

}