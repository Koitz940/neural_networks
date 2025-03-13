use rand::Rng;

#[derive(Clone, Debug)]
pub struct Sigmoid {
    pub weights: Vec<f64>,
    pub bias: f64,
}

fn vec_prod(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!()
    }
    std::iter::zip(a, b).fold(0.0, |acc, (i, j)| i.mul_add(*j, acc))
}

fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

impl Sigmoid {
    pub fn new(nweights: usize) -> Sigmoid {
        let mut rng = rand::rng();
        Sigmoid {
            weights: std::iter::repeat_with(|| rng.random_range(-1.0..1.0))
                .take(nweights)
                .collect(),
            bias: rng.random_range(-10.0..10.0),
        }
    }

    pub fn weight_sum(&self, input: &[f64]) -> f64 {
        sigma(vec_prod(&self.weights, input) + self.bias)
    }
}
