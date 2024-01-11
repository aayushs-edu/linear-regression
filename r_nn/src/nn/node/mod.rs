pub mod node {

    use ndarray::{Array1, ArrayView1};
    use std::fmt;

    /// calculates the dot product of two ndarray type objects
    pub fn dot_product(arr1: ArrayView1<f32>, arr2: ArrayView1<f32>) -> f32 {
        arr1.dot(&arr2)
    }

    #[derive(Clone)]
    pub struct Node {
        pub weights: Vec<f32>,
        pub bias: f32
    }

    impl fmt::Display for Node {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Weights: {:?}, bias: {}", self.weights, self.bias)
        }
    }

    impl Node {
        /// Creates a new Node Object with weights and a bias
        pub fn new(weights: Vec<f32>, bias: f32) -> Self {
            Node {
                weights,
                bias
            }
        }
        pub fn sigmoid_actualize(&self, features: Vec<f32>) -> f32 {
            let features_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(features);
            let weights_arr: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<_>, ndarray::prelude::Dim<[usize; 1]>> = Array1::from_vec(self.weights.clone());
            println!("calculating dot product of features and weights");
            let z: f32 = dot_product(features_arr.view(), weights_arr.view()) + self.bias;
            println!("done");
            let exp_z: f32 = (-z).exp();
            1.0 / (1.0 + exp_z)
        }
        pub fn set_weights(&mut self, weights: Vec<f32>, bias: f32) {
            self.weights = weights;
            self.bias = bias;
        }
        pub fn get_weights(&self) -> (Vec<f32>, f32) {
            (self.weights.clone(), self.bias)
        }
        pub fn str(&self) {
            println!("{}", &self)
        }
    }
}