use na::{DVector, DMatrix};
use rand::{Rng, seq::SliceRandom, thread_rng};
use rand_distr::StandardNormal;

use crate::data_item::DataItem;

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

        let (biases, weights) = 
            Network::init_weights_and_biases(&num_layers, &layer_sizes);

        Network {
            num_layers,
            layer_sizes,
            biases,
            weights
        }
    }
    
    pub fn feed_forward(&mut self, activation: DVector<f32>) -> DVector<f32> {
        self.feed_forward_record(activation, false).0
    }
    
    fn feed_forward_record(&mut self,
                           mut activation: DVector<f32>,
                           should_record: bool) -> (DVector<f32>, Vec<DVector<f32>>, Vec<DVector<f32>>) {
        let mut activation_history: Vec<DVector<f32>> = vec![activation.clone()];
        let mut z_history: Vec<DVector<f32>> = Vec::new();

        for (bias, weight) in self.biases.iter().zip(self.weights.iter()) {
            activation = weight * activation + bias;

            if should_record { z_history.push(activation.clone()); }

            Network::sigmoid(&mut activation);

            if should_record { activation_history.push(activation.clone()); }
        }

        (activation, activation_history, z_history)
    }

    pub fn sgd_learn(&mut self, 
                     training_data: &mut Vec<(DVector<f32>, DVector<f32>)>, 
                     num_epochs: usize, 
                     mini_batch_size: usize, 
                     learning_rate: f32, 
                     test_data: Option<&Vec<(DVector<f32>, DVector<f32>)>>) {
        for _ in 0..num_epochs {
            let mini_batches = Network::create_mini_batches(
                                 training_data, mini_batch_size);

            for batch in &mini_batches {
                self.update(batch, learning_rate);
            }
        }
    }
    
    fn update(&mut self, batch: &[(DVector<f32>, DVector<f32>)], learning_rate: f32) {
        let mut nabla_b: Vec<DVector<f32>> = Vec::new();
        let mut nabla_w: Vec<DMatrix<f32>> = Vec::new();

        for (input, output) in batch {
            let (nabla_b_shift, nabla_w_shift) = self.backprop(input, output);

            for (nb, nbs) in nabla_b.iter_mut().zip(nabla_b_shift.iter()) {
                *nb += nbs;
            }
            for (nw, nws) in nabla_w.iter_mut().zip(nabla_w_shift.iter()) {
                *nw += nws;
            }
        }

        for (w, nw) in self.weights.iter_mut().zip(nabla_w.iter()) {
            *w -= (learning_rate / batch.len() as f32) * nw;
        }
        for (b, nb) in self.biases.iter_mut().zip(nabla_b.iter()) {
            *b -= (learning_rate / batch.len() as f32) * nb;
        }
    }

    fn backprop(&mut self, input: &DVector<f32>, expected: &DVector<f32>) -> 
            (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let mut nabla_b: Vec<DVector<f32>> = Vec::new();
        let mut nabla_w: Vec<DMatrix<f32>> = Vec::new();

        let (_, activations, zs) = self.feed_forward_record(input.clone(), true);

        let mut last_z = match zs.last() {
            None => panic!("Neural network has no layers! Cannot backprop"),
            Some(e) => e.clone()
        };
        Network::sigmoid_prime(&mut last_z);

        let mut delta = 
            self.cost_derivative(activations.last().unwrap(), expected).component_mul(&last_z);

        nabla_b[self.num_layers - 1] = delta.clone();
        nabla_w[self.num_layers - 1] = 
            delta.clone() * activations[activations.len() - 2].transpose();

        for layer_num in 2..self.num_layers {
            let mut curr_z = zs[zs.len() - layer_num].clone();
            Network::sigmoid_prime(&mut curr_z);
            delta = self.weights[self.weights.len() - layer_num + 1].transpose() * delta;
            delta.component_mul_assign(&curr_z);
            nabla_b[self.num_layers - layer_num] = delta.clone();
            nabla_w[self.num_layers - layer_num] = 
                delta.clone() * activations[activations.len() - layer_num - 1].transpose();
        }

        (nabla_b, nabla_w)
    }

    fn cost_derivative(&self, 
                       output: &DVector<f32>, 
                       expected: &DVector<f32>) -> DVector<f32> {
        output - expected
    }

    fn create_mini_batches(training_data: &mut Vec<(DVector<f32>, DVector<f32>)>, 
                           mini_batch_size: usize) -> 
            Vec<&[(DVector<f32>, DVector<f32>)]> {
        let mut rng = rand::thread_rng();
        training_data.shuffle(&mut rng);

        let mut mini_batches: Vec<&[(DVector<f32>, DVector<f32>)]> = Vec::new();
        for batch_idx in (0..training_data.len()).step_by(mini_batch_size) {
            mini_batches.push(
                &training_data[batch_idx..batch_idx + mini_batch_size]
            );
        }

        mini_batches
    }


    fn init_weights_and_biases(num_layers: &usize, 
                               layer_sizes: &Vec<usize>) -> 
            (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let mut rng = thread_rng();
        
        let mut biases: Vec<DVector<f32>> = Vec::new();
        let mut weights: Vec<DMatrix<f32>> = Vec::new();

        let mut prev_layer_size = layer_sizes[0];
        for layer_idx in 1..*num_layers {
            let layer_size = layer_sizes[layer_idx];

            biases.push(DVector::from_fn(layer_size, |_r, _c| {
                rng.sample(StandardNormal)
            }));
            weights.push(DMatrix::from_fn(layer_size, prev_layer_size, |_r, _c| {
                rng.sample(StandardNormal)
            }));

            prev_layer_size = layer_size;
        }

        (biases, weights)
    }

    fn sigmoid(z: &mut DVector<f32>) {
        z.apply(|zi| *zi = 1.0 / (1.0 + (-*zi).exp()));
    }

    fn sigmoid_prime(z: &mut DVector<f32>) {
        z.apply(|zi| *zi = (-*zi).exp() / (1.0 + (-*zi).exp()).powf(2.0));
    }
}
