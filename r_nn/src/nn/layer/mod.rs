pub mod layer {
    use crate::nn::node::node::Node;
    use rand::Rng;
    use itertools::izip;

    pub struct Layer {
        pub nodes: Vec<Node>,
        pub num_nodes: usize
    }

    impl Layer {
        
        /// Creates a new Layer object with a given number of nodes
        pub fn new(num_nodes: usize) -> Layer {
            let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
            let weights: Vec<f32> = vec![rng.gen::<f32>()];
            let bias: f32 = rng.gen::<f32>();
            let nodes: Vec<Node> = (0..num_nodes).map(|_| Node::new(weights.clone(), bias)).collect();
            Layer {
                nodes,
                num_nodes
            }
        }
        pub fn set_weights(&mut self, index: usize, weights: Vec<f32>, bias: f32) {
            self.nodes[index].set_weights(weights, bias)
        }
        pub fn set_all_weights(&mut self, weights: Vec<Vec<f32>>, biases: Vec<f32>) {
            if weights.len() != biases.len() {
                panic!("Weights and bias lists must be the same length!")
            }
            for i in 0..self.num_nodes {
                self.nodes[i].set_weights(weights[i].clone(), biases[i]);
            }
        }
        pub fn execute(&self, input_layer: Vec<f32>) -> Vec<f32> {
            let num_features: usize = input_layer.len();
            let mut a_vector: Vec<f32> = Vec::new();
            for node in &self.nodes {
                if num_features != node.weights.len() {
                    panic!("Number of weights does not match number of features");
                }
                a_vector.push(node.sigmoid_actualize(input_layer.clone()));
            }
            a_vector
        }
        pub fn print_layer(&self) {
            for i in 0..self.num_nodes {
                println!("Node {}: {}", i + 1, self.nodes[i])
            }
        }
        pub fn get_nodes(&self) -> Vec<Node> {
            self.nodes.clone()
        }
    }
}