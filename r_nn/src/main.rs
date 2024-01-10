mod nn {
    pub mod layer;
    pub mod nn;
}

use nn::layer::layer::Layer;
use nn::nn::nn::NeuralNetwork;

fn main() {

    // weights and biases
    let w_1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.6],
        vec![0.7, 0.8]
    ];

    let w_2: Vec<Vec<f32>>

    let layer_1 = Layer::new(2);
    layer_1.set_all_weights(w_1)
}
