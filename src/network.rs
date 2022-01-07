use na::{DVector, DMatrix};
use rand::{Rng, thread_rng};
use rand_distr::{Distribution, StandardNormal};

pub struct Network {
    num_layers: usize,
    layer_sizes: Vec<usize>,
    biases: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
}

impl Network {
    pub fn new(num_layers: usize, layer_sizes: Vec<usize>) -> Network {
        assert_eq!(num_layers, layer_sizes.len(),
                   "number of layers does not match number of layer sizes!");

        let mut rng = thread_rng();
        
        let mut biases: Vec<DVector<f32>> = Vec::new();
        let mut weights: Vec<DMatrix<f32>> = Vec::new();
        for &layer_size in &layer_sizes {
            biases.push(DVector::from_fn(num_layers, |_r, _c| {
                rng.sample(StandardNormal)
            }));
            weights.push(DMatrix::from_fn(num_layers, layer_size, |_r, _c| {
                rng.sample(StandardNormal)
            }));
        }

        Network {
            num_layers,
            layer_sizes,
            biases,
            weights
        }
    }
}
