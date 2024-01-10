pub mod nn {
    use crate::nn::layer::layer::Layer;

    pub struct NeuralNetwork {
        pub layers: Vec<Layer>,
        pub num_layers: usize
    }

    impl NeuralNetwork {

        /// Creates a new Neural Network object with an input Vec<Layer>
        pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
            let num_layers: usize = layers.len();
            NeuralNetwork {
                layers,
                num_layers
            }
        }
        pub fn add_layer(&mut self, layer: Layer, location: usize) {
            self.layers.insert(location, layer);
            self.num_layers += 1;
        }
        pub fn predict(&self, input_layer: Vec<f32>) {
            let mut outputs: Vec<f32> = Vec::new();
            let input: Vec<f32> = input_layer;
            for layer in &self.layers {
                let output: Vec<f32> = layer.get_nodes().iter().map(|node| node.sigmoid_actualize(input.clone())).collect();
                outputs.push(output)

            }
        }
    }
}