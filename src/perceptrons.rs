pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

fn vec_prod(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!()
    }
    std::iter::zip(a, b).fold(0.0, |acc, (i, j)| i.mul_add(*j, acc))
}

impl Perceptron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Perceptron {
        Perceptron { weights, bias }
    }

    pub fn output(&self, input: &[f64]) -> u8 {
        match vec_prod(&self.weights, input) + self.bias > 0.0 {
            true => 1,
            false => 0,
        }
    }
}
