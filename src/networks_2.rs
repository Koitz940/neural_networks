use crate::mat_op;
use rand::Rng;
use rand_distr::StandardNormal;
use std::iter::{once, repeat_with, zip};

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Box<[Box<[f64]>]>,
    pub biases: Box<[f64]>,
}

impl Layer {
    pub fn new(amount_neurons: usize, amount_weights: usize) -> Layer {
        let mut rng = rand::rng();
        let weights = repeat_with(|| {
            (&mut rng)
                .sample_iter(StandardNormal)
                .take(amount_weights)
                .collect()
        })
        .take(amount_neurons)
        .collect();
        let biases = (&mut rng)
            .sample_iter(StandardNormal)
            .take(amount_neurons)
            .collect();
        Layer { weights, biases }
    }

    pub fn amount_neurons(&self) -> usize {
        self.biases.len()
    }

    pub fn amount_outputs(&self) -> usize {
        self.weights.len()
    }

    pub fn output(&self, input: &[f64]) -> Box<[f64]> {
        let n = mat_op::mat_vec_prod(&self.weights, input);
        mat_op::vec_sum(&n, &self.biases)
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, sizes: Vec<usize>) -> NeuralNetwork {
        let layers = zip(once(&input_size).chain(&sizes), &sizes)
            .map(|(weights, size)| Layer::new(*size, *weights))
            .collect();
        NeuralNetwork { layers }
    }

    pub fn output(&self, input: &Box<[f64]>) -> Box<[f64]> {
        let mut current = (*input).clone();
        for layer in &self.layers {
            current = layer.output(&current)
        }
        current
    }

    pub fn update_one(&mut self, x: &[f64], expected: &[f64]) {
        let mut z_s = Vec::new();
        let mut a_s = Vec::new();
        let mut current = x;
        for layer in &self.layers {
            let z = layer.output(current);
            let a = mat_op::vec_sigmoid(&z);
            z_s.push(z);
            a_s.push(a);
            current = a_s.last().unwrap(); //I just added an element, it should exist
        }
        let mut errors = Vec::new();
        let error = mat_op::vec_elementwise_prod(
            &mat_op::vec_sub(a_s.last().unwrap(), expected),
            &mat_op::vec_dsigmoid(z_s.last().unwrap()),
        );
        errors.push(error);
        if self.layers.len() > 1 {
            for (layer, z) in zip(&self.layers, z_s).rev().skip(1) {
                let error = mat_op::vec_elementwise_prod(
                    &mat_op::mat_vec_prod(&layer.weights, errors.last().unwrap()),
                    &mat_op::vec_dsigmoid(&z),
                );
                errors.push(error)
            }
        }
        errors.reverse();
        for i in 0..self.layers.len() {
            self.layers[i].weights = mat_op::mat_sub(
                &self.layers[i].weights,
                &mat_op::vec_vec_prod(&errors[i], &a_s[i]),
            );
            self.layers[i].biases = mat_op::vec_sub(&self.layers[i].biases, &errors[i])
        }
    }

    pub fn update_bunch(&mut self) {}
}
