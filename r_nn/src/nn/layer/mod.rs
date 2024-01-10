pub mod layer {
    use crate::nn::node::node::Node;

    pub struct Layer {
        pub nodes: Vec<Node>,
        pub num_nodes: i32
    }

    impl Layer {
        pub fn new(num_nodes: i32) -> Layer {
            
        }
    }
}