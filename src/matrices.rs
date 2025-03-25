use std::f64::consts::E;
use std::iter::{repeat_with, zip};

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn dsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn vec_prod(a: &[f64], b: &[f64]) -> f64 {
    zip(a, b).fold(0.0, |acc, (i, j)| i.mul_add(*j, acc))
}

pub fn vec_sum(a: &[f64], b: &[f64]) -> Vec<f64> {
    zip(a, b).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    zip(a, b).map(|(x, y)| x - y).collect()
}

pub fn mat_prod(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|col| transposed(b).iter().map(|row| vec_prod(col, row)).collect())
        .collect()
}

pub fn transposed(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..mat[0].len())
        .map(|j| (0..mat.len()).map(|i| mat[i][j]).collect())
        .collect()
}

#[derive(Clone, Debug)]
pub struct Matrix {
    rows: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn rows(&self) -> std::slice::Iter<'_, std::vec::Vec<f64>> {
        self.rows.iter()
    }
    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    pub fn ncols(&self) -> usize {
        self.rows[0].len()
    }

    pub fn dimensions(&self) {
        println!("{} {}", self.nrows(), self.ncols())
    }
    pub fn new(rows: &[Vec<f64>]) -> Result<Matrix, &str> {
        match rows.is_empty() {
            true => Err("matrix cannot be empty"),
            false => match rows[0].len() {
                0 => Err("rows cannot be empty"),
                _ => match rows.iter().all(|i| rows[0].len() == i.len()) {
                    false => Err("all rows must be the same length"),
                    true => Ok(Matrix {
                        rows: rows.to_vec(),
                    }),
                },
            },
        }
    }

    pub fn into_matrix<'a>(rows: Vec<Vec<f64>>) -> Result<Matrix, &'a str> {
        match rows.is_empty() {
            true => Err("matrix cannot be empty"),
            false => match rows[0].len() {
                0 => Err("rows cannot be empty"),
                _ => match rows.iter().all(|i| rows[0].len() == i.len()) {
                    false => Err("all rows must be the same length"),
                    true => Ok(Matrix { rows }),
                },
            },
        }
    }

    pub fn sum<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.rows.len() == b.rows.len() && a.rows[0].len() == b.rows[0].len() {
            Ok(Matrix {
                rows: zip(&a.rows, &b.rows).map(|(x, y)| vec_sum(x, y)).collect(),
            })
        } else {
            Err("in order to sum matrices, they must have the same dimensions")
        }
    }

    pub fn sub<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.rows.len() == b.rows.len() && a.rows[0].len() == b.rows[0].len() {
            Ok(Matrix {
                rows: zip(&a.rows, &b.rows).map(|(x, y)| vec_sub(x, y)).collect(),
            })
        } else {
            Err("in order to subtract matrices, they must have the same dimensions")
        }
    }

    pub fn sum_inplace(&mut self, a: &Matrix) {
        if a.rows[0].len() == self.rows[0].len() && self.rows.len() == a.rows.len() {
            for i in 0..self.rows.len() {
                for j in 0..self.rows[i].len() {
                    self.rows[i][j] += a.rows[i][j]
                }
            }
        } else {
            panic!()
        }
    }

    pub fn sub_inplace(&mut self, a: &Matrix) {
        if a.rows[0].len() == self.rows[0].len() && self.rows.len() == a.rows.len() {
            for i in 0..self.rows.len() {
                for j in 0..self.rows[i].len() {
                    self.rows[i][j] -= a.rows[i][j]
                }
            }
        } else {
            panic!()
        }
    }

    pub fn mult<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.rows[0].len() == b.rows.len() {
            Ok(Matrix {
                rows: mat_prod(&a.rows, &b.rows),
            })
        } else {
            Err("in order to multiply 2 matrices, the length of the rows of the first one must be equal to the amount of rows of the second one")
        }
    }

    pub fn mult_inplace(&mut self, a: &Matrix) {
        assert!(self.rows[0].len() == a.rows.len());
        self.rows = mat_prod(&self.rows, &a.rows)
    }

    pub fn transposed(&self) -> Matrix {
        Matrix {
            rows: transposed(&self.rows),
        }
    }

    pub fn transpose(&mut self) {
        self.rows = self.transposed().rows
    }

    pub fn sigmoid(&self) -> Matrix {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(|x| sigmoid(*x)).collect())
            .collect();
        Matrix { rows }
    }

    pub fn dsigmoid(&self) -> Matrix {
        let rows = self
            .rows
            .iter()
            .map(|row| row.iter().map(|x| dsigmoid(*x)).collect())
            .collect();
        Matrix { rows }
    }

    pub fn elementwise_mult(&self, a: &Matrix) -> Matrix {
        let rows = zip(&self.rows, &a.rows)
            .map(|(row1, row2)| zip(row1, row2).map(|(x, y)| x * y).collect())
            .collect();
        Matrix { rows }
    }

    pub fn cons_prod(&self, x: &f64) -> Matrix {
        let rows: Vec<Vec<f64>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(|y| y * x).collect())
            .collect();
        Matrix { rows }
    }

    pub fn sum_rows(&self) -> Matrix {
        let mut iter_rows = self.rows.iter().map(|x| Matrix::new(&[x.clone()]).unwrap());
        let init = iter_rows.next().unwrap();
        iter_rows.fold(init, |acc, x| &acc + &x)
    }

    pub fn extendedcol(&self, n: usize) -> Matrix {
        Matrix::into_matrix(
            self.rows()
                .map(|row| repeat_with(|| row[0]).take(n).collect())
                .collect(),
        )
        .unwrap()
    }

    pub fn extendedrow(&self, n: usize) -> Matrix {
        Matrix::into_matrix(repeat_with(|| self.rows[0].clone()).take(n).collect()).unwrap()
    }
}

impl std::ops::Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix::sum(self, rhs).unwrap()
    }
}

impl std::ops::AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        self.sum_inplace(&rhs);
    }
}

impl std::ops::Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix::mult(self, rhs).unwrap()
    }
}

impl std::ops::MulAssign for Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        self.mult_inplace(&rhs);
    }
}

impl std::ops::Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix::sub(self, rhs).unwrap()
    }
}

impl std::ops::SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_inplace(&rhs);
    }
}
