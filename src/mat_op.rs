use std::f64::consts::E;
use std::iter::{once, repeat_with, zip};

pub fn scal_vec_prod(x: &f64, a: &[f64]) -> Box<[f64]> {
    a.iter().map(|c| x * c).collect()
}

pub fn scal_mat_prod(x: &f64, a: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    a.iter().map(|c| scal_vec_prod(x, c)).collect()
}

pub fn vec_prod(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        panic!()
    }
    zip(a, b).fold(0.0, |acc, (i, j)| i.mul_add(*j, acc))
}

pub fn mat_vec_prod(mat: &[Box<[f64]>], vector: &[f64]) -> Box<[f64]> {
    if !mat.iter().all(|row| row.len() == vector.len()) {
        panic!()
    }
    mat.iter().map(|a| vec_prod(a, vector)).collect()
}

pub fn transposed(mat: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    match mat.len() {
        0 => Vec::new().into_boxed_slice(),
        _ => match mat.iter().all(|row| row.len() == mat[0].len()) {
            true => (0..mat[0].len())
                .map(|i| mat.iter().map(|a| a[i]).collect())
                .collect(),
            false => panic!(),
        },
    }
}

pub fn vec_sum(a: &[f64], b: &[f64]) -> Box<[f64]> {
    if a.len() != b.len() {
        panic!()
    };
    zip(a, b).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Box<[f64]> {
    if a.len() != b.len() {
        panic!()
    };
    zip(a, b).map(|(x, y)| x - y).collect()
}

pub fn mat_sub(a: &[Box<[f64]>], b: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    zip(a, b).map(|(x, y)| vec_sub(x, y)).collect()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn dsigmoid(x: f64) -> f64 {
    sigmoid(x) / (1.0 - sigmoid(x))
}

pub fn vec_sigmoid(x: &[f64]) -> Box<[f64]> {
    x.iter().map(|a| sigmoid(*a)).collect()
}

pub fn vec_dsigmoid(x: &[f64]) -> Box<[f64]> {
    x.iter().map(|a| dsigmoid(*a)).collect()
}

pub fn vec_elementwise_prod(x: &[f64], y: &[f64]) -> Box<[f64]> {
    if x.len() != y.len() {
        panic!();
    }
    zip(x, y).map(|(a, b)| a * b).collect()
}

pub fn vec_vec_prod(x: &[f64], y: &[f64]) -> Box<[Box<[f64]>]> {
    x.iter()
        .map(|a| y.iter().map(|b| a * b).collect())
        .collect()
}

pub fn mat_prod(a: &[Box<[f64]>], b: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    if a.last().unwrap().len() != b.len() {
        panic!()
    }
    a.iter()
        .map(|col| transposed(b).iter().map(|row| vec_prod(col, row)).collect())
        .collect()
}
