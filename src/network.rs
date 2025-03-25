use crate::matrices::Matrix;
use itertools::izip;
use rand::Rng;
use rand_distr::StandardNormal;
use std::iter::{once, repeat_with, zip};

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
}

impl Layer {
    pub fn new(amount_neurons: usize, amount_weights: usize) -> Layer {
        let mut rng = rand::rng();
        let rows = repeat_with(|| {
            repeat_with(|| rng.sample::<f64, _>(StandardNormal))
                .take(amount_neurons)
                .collect()
        })
        .take(amount_weights)
        .collect();
        let weights = Matrix::into_matrix(rows).unwrap();
        let rows = vec![repeat_with(|| rng.sample::<f64, _>(StandardNormal))
            .take(amount_neurons)
            .collect()];
        let biases = Matrix::into_matrix(rows).unwrap();
        Layer { weights, biases }
    }

    pub fn output(&self, input: &Matrix) -> Matrix {
        Matrix::sigmoid(&(&self.biases + &(input * &self.weights)))
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub learn_rate: f64,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, sizes: Vec<usize>, learn_rate: f64) -> NeuralNetwork {
        let layers = zip(once(&input_size).chain(&sizes), &sizes)
            .map(|(weights, size)| Layer::new(*size, *weights))
            .collect();
        NeuralNetwork { layers, learn_rate }
    }

    pub fn output(&self, input: &Matrix) -> Matrix {
        let mut layer_iter = self.layers.iter();
        let init = layer_iter.next().unwrap().output(input);
        layer_iter.fold(init, |acc, x| x.output(&acc))
    }

    pub fn learn(&mut self, input: &Matrix, expected: &Matrix) {
        let mut z_s = vec![input.clone()];
        let mut a_s = vec![input.clone()];
        for layer in &self.layers {
            let c = a_s.last().unwrap() * &layer.weights;
            assert!(!c.rows().last().unwrap()[0].is_nan(), "c");
            let z = &layer.biases.extendedrow(c.nrows()) + &c;
            let a = Matrix::sigmoid(&z);
            a_s.push(a);
            z_s.push(z);
        }
        let error = a_s.last().unwrap() - expected;
        let mut errors = vec![error];
        z_s.pop();
        a_s.pop();
        for (layer, z) in zip(&self.layers, z_s).rev().take(self.layers.len() - 1) {
            errors.push(Matrix::elementwise_mult(
                &(errors.last().unwrap() * &layer.weights.transposed()),
                &Matrix::dsigmoid(&z),
            ));
        }
        errors.reverse();

        for (layer, error, a) in izip!(&mut self.layers, errors, a_s) {
            let combined = &a.transposed() * &error;
            layer.weights +=
                Matrix::cons_prod(&combined, &(self.learn_rate / (input.nrows() as f64)));
            let avg_error = Matrix::into_matrix(vec![error
                .transposed()
                .rows()
                .map(|x| x.iter().sum())
                .collect()])
            .unwrap();
            layer.biases +=
                Matrix::cons_prod(&avg_error, &(self.learn_rate / (input.nrows() as f64)));
        }
    }
    pub fn check_one(&self, input: Matrix, expected: Matrix) -> f64 {
        let out = self.output(&input);
        let v = out.rows().next().unwrap().to_vec();
        let e = expected.rows().next().unwrap().to_vec();
        if e.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .unwrap()
            .0
            == v.iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap()
                .0
        {
            1.0
        } else {
            0.0
        }
    }
}
