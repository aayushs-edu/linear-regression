pub mod nn {
    use core::num;

    use crate::nn::layer::layer::Layer;

    pub struct NeuralNetwork {
        pub layers: Vec<Layer>,
        pub num_layers: usize
    }

    impl NeuralNetwork {

        /// Creates a new Neural Network object with an input Vec<Layer>
        pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
            let num_layers: usize = layers.len();
            println!("CREATED NEURAL NETWORK OBJECT WITH NUM_LAYERS: {}", num_layers);
            NeuralNetwork {
                layers,
                num_layers
            }
        }
        pub fn add_layer(&mut self, layer: Layer, location: usize) {
            self.layers.insert(location, layer);
            self.num_layers += 1;
        }
        pub fn predict(&self, input_layer: Vec<f32>) -> f32 {
            let mut outputs: Vec<f32> = Vec::new();
            let input: Vec<f32> = input_layer;
            let mut index: i32 = 1;
            for layer in &self.layers {
                println!("activating nodes in hidden layer {}", index);
                index += 1;
                let output: Vec<f32> = layer.get_nodes().iter().map(|node| node.sigmoid_actualize(input.clone())).collect();
                outputs.extend(output.clone())
            }
            println!("{:?}", outputs);
            return outputs[outputs.len() - 1];
        }
        pub fn print_layers(&self) {
            for i in 0..self.num_layers {
                println!("Layer {}", i + 1);
                self.layers[i].print_layer();

            }
        }
    }
}