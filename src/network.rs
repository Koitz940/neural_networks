use crate::sigmoid::Sigmoid;
use std::iter::{once, repeat_with, zip};

#[derive(Debug)]
pub struct Network {
    pub layers: Box<[Box<[Sigmoid]>]>,
}

impl Network {
    pub fn new(input_size: usize, size: Vec<usize>) -> Network {
        let layers = zip(once(&input_size).chain(&size), &size)
            .map(|(weights, size)| repeat_with(|| Sigmoid::new(*weights)).take(*size).collect())
            .collect();
        Network { layers }
    }
    pub fn input_size(&self) -> usize {
        match self.layers.as_ref() {
            [first_layer, ..] => match first_layer.as_ref() {
                [first_sigmoid, ..] => first_sigmoid.weights.len(),
                _ => 0,
            },
            _ => 0,
        }
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().map(|s| s.len()).unwrap_or_default()
    }

    pub fn output(&self, input: &[f64]) -> Vec<f64> {
        if input.len() != self.input_size() {
            panic!()
        }
        let mut new_input: Vec<f64> = Vec::from(input);
        for layer in &self.layers {
            new_input = layer
                .iter()
                .map(|sigmoid| sigmoid.weight_sum(&new_input))
                .collect::<Vec<f64>>();
        }
        new_input
    }
    pub fn cost(&self, input: &[f64], expected: &[f64]) {
        let out = self.output(input);
        if out.len() != expected.len() {
            panic!()
        }
    }
}
