mod nn {
    pub mod layer;
    pub mod nn;
    pub mod node;
}

mod gradient_descent {
    pub mod obj;
    pub mod stochastic;
    pub mod batch;
}

mod adam;

use core::num;

use nn::layer::layer::Layer;
use nn::nn::nn::NeuralNetwork;
use backtrace::Backtrace;
use gradient_descent::obj::obj::GradientDescent;
use gradient_descent::stochastic::stochastic::stochastic_vect;
use gradient_descent::batch::batch::{batch, batch_vectorized};
use adam::adam::Adam;
use std::fs::{File, OpenOptions};
use std::io::{Write, Error};

fn file_save(data: String) -> Result<(), Error> {
    let file_path = "C:/Users/ellio/OneDrive/Documents/GitHub/linear-regression/log.txt";
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;
    file.write_all(data.as_bytes())?;
    file.write(b"\n\n")?;
    Ok(())
}


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
        vec![0.05, 0.2]
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
    let mut i: i8 = 1;
    for layer in layers.clone() {
        println!("LAYER: {}", i);
        i += 1;
        layer.print_layer()
    }

    let nn: NeuralNetwork = NeuralNetwork::new(layers);
    let input_layer: Vec<f32> = vec![0.0, 0.0];
    let prediction: f32 = nn.predict(input_layer);
    println!("\n");
    println!("PREDICTION: {}", prediction);
    println!("\n");

    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_train: Vec<f64> = vec![2.0, 4.0, 5.0, 4.5, 5.5];
    let learning_rate: f64 = 0.01;
    let num_predictors: usize = x_train.len();
    let gd: GradientDescent = GradientDescent::new(
        x_train.clone(),
        y_train.clone(),
        num_predictors,
        learning_rate
    );

    let stepsize: f64 = 0.001;
    let beta_1: f64 = 0.9;
    let beta_2: f64 = 0.009;
    let epsilon: f64 = 1e-8;
    let default: bool = true;
    let gd_clone: GradientDescent = gd.clone();
    let mut adam: Adam = Adam::new(
        gd_clone,
        default,
        stepsize,
        beta_1,
        beta_2,
        epsilon
    );

    let epochs: usize = 100;
    adam.optimize(epochs);
    println!("\n\n\n");
    println!("Final Weights: {:?}", gd.get_params());
    println!("Final Bias: {:?}", gd.get_y());
    println!("\n\n\n");

    // log
    let mut log_data: String = String::new();
    log_data.push_str("Final Weights: ");
    log_data.push_str(&format!("{:?}", gd.get_params()));
    log_data.push_str("\nFinal Bias: ");
    log_data.push_str(&format!("{:?}", gd.get_y()));
    file_save(log_data);

    // BACKTRACE LOG
    println!("{:?}", bt);

}
