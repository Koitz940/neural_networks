use crate::mat_op;
use itertools::izip;
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
    pub learn_rate: f64,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, sizes: Vec<usize>) -> NeuralNetwork {
        let layers = zip(once(&input_size).chain(&sizes), &sizes)
            .map(|(weights, size)| Layer::new(*size, *weights))
            .collect();
        NeuralNetwork {
            layers,
            learn_rate: -0.25,
        }
    }

    pub fn output(&self, input: &[f64]) -> Box<[f64]> {
        let mut layer_iter = self.layers.iter();
        let init = layer_iter.next().unwrap().output(input);
        layer_iter.fold(init, |acc, x| x.output(&acc))
    }

    pub fn update_one(&mut self, x: &[f64], expected: &[f64]) {
        let mut z_s = Vec::new();
        let mut a_s = Vec::new();
        let mut current = x;
        a_s.push(x.iter().copied().collect());
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
        for (layer, z) in zip(&self.layers, z_s).rev().skip(1) {
            let error = mat_op::vec_elementwise_prod(
                &mat_op::mat_vec_prod(&layer.weights, errors.last().unwrap()),
                &mat_op::vec_dsigmoid(&z),
            );
            errors.push(error);
        }
        errors.reverse();
        a_s.pop();
        for (layer, a, error) in izip!(&mut self.layers, a_s, errors) {
            layer.weights = mat_op::mat_sum(
                &layer.weights,
                &mat_op::scal_mat_prod(&self.learn_rate, &mat_op::vec_vec_prod(&error, &a)),
            );
            layer.biases = mat_op::vec_sum(
                &layer.biases,
                &mat_op::scal_vec_prod(&self.learn_rate, &error),
            );
        }
    }

    pub fn test_one(&mut self, x: &[f64], expected: &[f64]) -> f64 {
        if expected
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            == self
                .output(x)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
        {
            1.0
        } else {
            0.0
        }
    }

    pub fn sample_lerning(&mut self, x: &[Box<[f64]>], expected: &[Box<[f64]>]) {
        let m = x.len() as f64;
        let mut z_s = Vec::new();
        let mut a_s = Vec::new();
        let mut current = x;
        for layer in &self.layers {
            let z: Box<[Box<[f64]>]> = mat_op::mat_tprod(&layer.weights, current)
                .iter()
                .map(|a| mat_op::vec_sum(a, &layer.biases))
                .collect();
            let a = mat_op::mat_sigmoid(&z);
            z_s.push(z);
            a_s.push(a);
            current = a_s.last().unwrap(); //I just added an element, it should exist
        }
        let mut errors = Vec::new();
        let error = mat_op::mat_elementwise_prod(
            &mat_op::mat_sub(a_s.last().unwrap(), expected),
            &mat_op::mat_dsigmoid(z_s.last().unwrap()),
        );
        errors.push(error);
        if self.layers.len() > 1 {
            for (layer, z) in zip(&self.layers, z_s).rev().skip(1) {
                let error = mat_op::mat_elementwise_prod(
                    &mat_op::mat_prod(&layer.weights, errors.last().unwrap()),
                    &mat_op::mat_dsigmoid(&z),
                );
                errors.push(error)
            }
        }
        errors.reverse();
        for i in 0..self.layers.len() {
            self.layers[i].weights = mat_op::mat_sum(
                &self.layers[i].weights,
                &mat_op::scal_mat_prod(
                    &(self.learn_rate / m),
                    &mat_op::vec_vec_prod(
                        &mat_op::mat_colsum(&errors[i]),
                        &mat_op::mat_colsum(&a_s[i]),
                    ),
                ),
            );
            self.layers[i].biases = mat_op::vec_sum(
                &self.layers[i].biases,
                &mat_op::scal_vec_prod(&(self.learn_rate / m), &mat_op::mat_colsum(&errors[i])),
            )
        }
    }
}
