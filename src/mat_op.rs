use std::f64::consts::E;
use std::iter::zip;

pub fn scal_vec_prod(x: &f64, a: &[f64]) -> Box<[f64]> {
    a.iter().map(|c| x * c).collect()
}

pub fn scal_mat_prod(x: &f64, a: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    a.iter().map(|c| scal_vec_prod(x, c)).collect()
}

pub fn vec_prod(a: &[f64], b: &[f64]) -> f64 {
    assert!(a.len() == b.len());
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
    assert!(a.len() == b.len());
    zip(a, b).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Box<[f64]> {
    assert!(a.len() == b.len());
    zip(a, b).map(|(x, y)| x - y).collect()
}

pub fn mat_sum(a: &[Box<[f64]>], b: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    zip(a, b).map(|(x, y)| vec_sum(x, y)).collect()
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

pub fn mat_sigmoid(x: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    x.iter().map(|a| vec_sigmoid(a)).collect()
}

pub fn mat_dsigmoid(x: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    x.iter().map(|a| vec_dsigmoid(a)).collect()
}

pub fn vec_elementwise_prod(x: &[f64], y: &[f64]) -> Box<[f64]> {
    assert!(x.len() == y.len());
    zip(x, y).map(|(a, b)| a * b).collect()
}

pub fn mat_elementwise_prod(x: &[Box<[f64]>], y: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    assert!(zip(x, y).all(|(a, b)| a.len() == b.len()));
    zip(x, y).map(|(a, b)| vec_elementwise_prod(a, b)).collect()
}

pub fn vec_vec_prod(x: &[f64], y: &[f64]) -> Box<[Box<[f64]>]> {
    x.iter()
        .map(|a| y.iter().map(|b| a * b).collect())
        .collect()
}

pub fn mat_prod(a: &[Box<[f64]>], b: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    assert!(a.last().unwrap().len() != b.len());
    a.iter()
        .map(|col| transposed(b).iter().map(|row| vec_prod(col, row)).collect())
        .collect()
}

pub fn mat_tprod(a: &[Box<[f64]>], b: &[Box<[f64]>]) -> Box<[Box<[f64]>]> {
    assert!(a.last().unwrap().len() != b.len());
    a.iter()
        .map(|col| b.iter().map(|row| vec_prod(col, row)).collect())
        .collect()
}

pub fn mat_colsum(x: &[Box<[f64]>]) -> Box<[f64]> {
    transposed(x).iter().map(|a| a.iter().sum()).collect()
}
