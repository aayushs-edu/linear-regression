pub mod adam {

    /// Adam optimization algorithm implementation
    /// Src: https://arxiv.org/abs/1412.6980

    use crate::gradient_descent::obj::obj::GradientDescent;
    use crate::batch;
    use crate::batch_vectorized;
    use crate::stochastic_vect;
    
    pub struct Adam {
        gd: GradientDescent,
        stepsize: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64
    }

    impl Adam {

        /// Creates a new Adam optimizer object. If default is passed as true,
        /// then the base coefficients for the stepsize and exponential decay rates are set to
        /// default, being alpha = 0.0001, beta 1 = 0.9, beta 2 = 0.999, and epsilon = 1e-8
        /// When not using default coefficients, the new function can intake values for 
        /// each of these parameters
        ///
        /// ### Parameters
        /// 
        /// a: stepsize
        /// 
        /// beta_1 & beta_2: exponential decay rates for the moment estimates
        /// 
        /// epsilon: small constant used to avoid division by zero when calculating RMS of the gradients
        pub fn new(gd: GradientDescent, default: bool, stepsize: f64, beta_1: f64, beta_2: f64, epsilon: f64) -> Adam {
            if default {
                let stepsize: f64 = 0.0001;
                let beta_1: f64 = 0.9;
                let beta_2: f64 = 0.999;
                let epsilon: f64 = 1e-8;
                println!("Loading default Adam parameters: alpha: {}, beta 1: {}, beta 2: {}, epsilon: {}", stepsize, beta_1, beta_2, epsilon);
                Adam {
                    gd,
                    stepsize,
                    beta_1,
                    beta_2,
                    epsilon
                }
            } else {
                println!("Using specified Adam parameters: alpha: {}, beta 1: {}, beta 2: {}, epsilon: {}", stepsize, beta_1, beta_2, epsilon);
                Adam {
                    gd,
                    stepsize,
                    beta_1,
                    beta_2,
                    epsilon
                }
            }
        }
    }
}