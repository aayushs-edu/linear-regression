mod nn {
    pub mod layer;
    pub mod nn;
    pub mod node;
}

use nn::layer::layer::Layer;
use nn::nn::nn::NeuralNetwork;
use backtrace::Backtrace;

fn main() {

    // BACKTRACE
    let bt = Backtrace::new();

    // weights and biases
    let w_1: Vec<Vec<f32>> = vec![
        vec![0.5, 0.6],
        vec![0.7, 0.8]
    ];

    let w_2: Vec<Vec<f32>> = vec![
        vec![0.7, 0.23],
        vec![0.1, 0.4],
        vec![0.3, 1.5]
    ];

    let w_3: Vec<Vec<f32>> = vec![
        vec![0.05, 0.2, 0.5]
    ];

    let b_1: Vec<f32> = vec![
        0.07, 0.05
    ];

    let b_2: Vec<f32> = vec![
        0.2, 0.5, 0.01
    ];

    let b_3: Vec<f32> = vec![
        0.03
    ];

    let mut layer_1: Layer = Layer::new(2);
    layer_1.set_all_weights(w_1, b_1);
    let mut layer_2: Layer = Layer::new(3);
    layer_2.set_all_weights(w_2, b_2);
    let mut layer_3: Layer = Layer::new(1);
    layer_3.set_all_weights(w_3, b_3);

    let layers: Vec<Layer> = vec![layer_1, layer_2, layer_3];
    let nn: NeuralNetwork = NeuralNetwork::new(layers);
    let input_layer: Vec<f32> = vec![0.0, 0.0];
    let prediction = nn.predict(input_layer);
    println!("{}", prediction);

    // BACKTRACE LOG
    println!("{:?}", bt);

}
