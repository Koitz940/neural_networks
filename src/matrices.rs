use std::f64::consts::E;
use std::iter::zip;




pub fn num_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn num_dsigmoid(x: f64) -> f64 {
    num_sigmoid(x) * (1.0 - num_sigmoid(x))
}

#[derive(Clone, Debug)]
pub struct Matrix {
    buffer: Vec<f64>,
    row_size: usize,
    col_size: usize
}

impl Matrix {
    pub fn nrows(&self) -> usize {
        self.col_size
    }

    pub fn ncols(&self) -> usize {
        self.row_size
    }

    pub fn dimensions(&self) {
        println!("{} {}", self.col_size, self.row_size)
    }
    pub fn new(buffer: &[f64], row_size: usize, col_size: usize) -> Result<Matrix, &str> {
        match buffer.is_empty() | (row_size == 0) | (col_size == 0) {
            true => Err("matrix cannot be empty"),
            false => match row_size*col_size == buffer.len(){
                true => Ok(Matrix{
                    row_size,
                    col_size,
                    buffer: buffer.to_vec()
                }),
                false => Err("Mismatched matrix dimensions")
            },
        }
    }

    pub fn into_matrix<'a>(buffer: Vec<f64>, row_size: usize, col_size: usize) -> Result<Matrix, &'a str> {
        match buffer.is_empty() | (row_size == 0) | (col_size == 0) {
            true => Err("matrix cannot be empty"),
            false => match row_size*col_size == buffer.len(){
                true => Ok(Matrix{
                    row_size,
                    col_size,
                    buffer
                }),
                false => Err("Mismatched matrix dimensions")
            },
        }
    }

    pub fn sum<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.row_size == b.row_size && a.col_size == b.col_size{
            Ok(Matrix{
            row_size: a.row_size,
            col_size: a.col_size,
            buffer: zip(&a.buffer, &b.buffer).map(|(x, y)| x + y).collect()
        })
        }
        else{Err("Matricies cannot be summed as they don't have the same dimensions")}

    }

    pub fn sub<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.row_size == b.row_size && a.col_size == b.col_size{
            Ok(Matrix{
            row_size: a.row_size,
            col_size: a.col_size,
            buffer: zip(&a.buffer, &b.buffer).map(|(x, y)| x - y).collect()
        })
        }
        else{Err("Matricies cannot be subtracted as they don't have the same dimensions")}
    }

    pub fn sum_inplace(&mut self, a: &Matrix) {
        if a.row_size == self.row_size && a.col_size == self.col_size {
            for i in 0..(self.row_size * self.col_size) {
                self.buffer[i] += a.buffer[i]
            }
        } else {
            panic!("Matricies cannot be summed as they don't have the same dimensions")
        }
    }

    pub fn sub_inplace(&mut self, a: &Matrix) {
        if a.row_size == self.row_size && a.col_size == self.col_size {
            for i in 0..(self.row_size * self.col_size) {
                self.buffer[i] -= a.buffer[i]
            }
        } else {
            panic!("Matricies cannot be subtracted as they don't have the same dimensions")
        }
    }

    pub fn mult<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.row_size == b.col_size{
            let mut buffer = Vec::with_capacity(a.col_size * b.row_size);
            for i in 0..a.col_size{
                for j in 0..b.row_size{
                    buffer.push((0..a.row_size).fold(0.0, |acc, ind| a.buffer[ind + a.row_size*i].mul_add(b.buffer[j + ind*b.row_size], acc)))
                }
            }
            Ok(Matrix{
                buffer,
                row_size: b.row_size,
                col_size: a.col_size
            })
        }
        else{Err("Matricies cannot be multiplied, amount of columns of the first one must match amount of rows of the second one (nxn)")}
    }

    pub fn t1mult<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.col_size == b.col_size{
            let mut buffer = Vec::with_capacity(a.row_size * b.row_size);
            for i in 0..a.row_size{
                for j in 0..b.row_size{
                    buffer.push((0..a.col_size).fold(0.0, |acc, ind| a.buffer[i + ind*a.row_size].mul_add(b.buffer[j + ind*b.row_size], acc)))
                }
            }
            Ok(Matrix{
                buffer,
                row_size: b.row_size,
                col_size: a.row_size
            })
        }
        else{Err("Matricies cannot be multiplied, amount of columns of the first one must match amount of rows of the second one (txn)")}
    }

    pub fn t2mult<'a>(a: &Matrix, b: &Matrix) -> Result<Matrix, &'a str> {
        if a.row_size == b.row_size{
            let mut buffer = Vec::with_capacity(a.col_size * b.col_size);
            for i in 0..a.col_size{
                for j in 0..b.col_size{
                    buffer.push((0..a.row_size).fold(0.0, |acc, ind| a.buffer[ind + a.row_size*i].mul_add(b.buffer[ind + b.row_size*j], acc)))
                }
            }
            Ok(Matrix{
                buffer,
                row_size: b.col_size,
                col_size: a.col_size
            })
        }
        else{Err("Matricies cannot be multiplied, amount of columns of the first one must match amount of rows of the second one (nxt)")}
    }

    pub fn mult_inplace(&mut self, a: &Matrix) {
        assert!(self.row_size == a.col_size, "Matricies cannot be multiplied, amount of columns of the first one: {} must match amount of rows of the second one: {}", self.ncols(), a.nrows());
        let mut buffer = Vec::with_capacity(self.col_size * a.row_size);
            for i in 0..self.col_size{
                for j in 0..a.row_size{
                    buffer.push((0..self.row_size).fold(0.0, |acc, ind| self.buffer[ind + self.row_size*i].mul_add(a.buffer[j + ind*a.row_size], acc)))
                }
            }
        self.buffer = buffer;
        self.row_size = a.row_size;
    }

    pub fn t1mult_inplace(&mut self, a: &Matrix) {
        assert!(self.col_size == a.col_size, "Matricies cannot be multiplied, amount of columns of the first one: {} must match amount of rows of the second one: {}", self.nrows(), a.nrows());
        let mut buffer = Vec::with_capacity(self.row_size * a.row_size);
            for i in 0..self.row_size{
                for j in 0..a.row_size{
                    buffer.push((0..self.col_size).fold(0.0, |acc, ind| self.buffer[i + ind*self.row_size].mul_add(a.buffer[j + ind*a.row_size], acc)))
                }
            }
        self.buffer = buffer;
        self.col_size = self.row_size;
        self.row_size = a.row_size;
    }

    pub fn t2mult_inplace(&mut self, a: &Matrix) {
        assert!(self.row_size == a.row_size, "Matricies cannot be multiplied, amount of columns of the first one: {} must match amount of rows of the second one: {}", self.ncols(), a.ncols());
        let mut buffer = Vec::with_capacity(self.col_size * a.col_size);
            for i in 0..self.col_size{
                for j in 0..a.col_size{
                    buffer.push((0..self.row_size).fold(0.0, |acc, ind| self.buffer[ind + self.row_size*i].mul_add(a.buffer[ind + a.row_size*j], acc)))
                }
            }
        self.buffer = buffer;
        self.row_size = a.col_size;
        
    }


    pub fn sigmoid(&self) -> Matrix {
        Matrix { 
            row_size: self.row_size,
            col_size: self.col_size,
            buffer: self.buffer.iter().map(|x| num_sigmoid(*x)).collect()
        }
    }

    pub fn dsigmoid(&self) -> Matrix {
        Matrix { 
            row_size: self.row_size,
            col_size: self.col_size,
            buffer: self.buffer.iter().map(|x| num_dsigmoid(*x)).collect()
        }
    }

    pub fn elementwise_mult(&self, a: &Matrix) -> Matrix {
        assert!(a.row_size == self.row_size && a.col_size == self.col_size);
        Matrix {
            row_size: self.row_size,
            col_size: self.col_size,
            buffer: zip(&self.buffer, &a.buffer).map(|(x, y)| x*y).collect()
        }
    }

    pub fn cons_prod(&mut self, x: f64) {
        for i in self.buffer.iter_mut(){
            *i *= x
        }
    }

    pub fn sum_all_rows(&mut self, other: &Matrix){
        assert!(other.row_size == self.row_size && other.col_size == 1);
        for i in 0..(self.nrows()*self.ncols()){
            self.buffer[i] += other.buffer[i % other.row_size]
        }
    }

    pub fn sum_of_cols(&self) -> Matrix{
        let mut buffer = Vec::with_capacity(self.row_size);
        for i in 0..self.ncols(){
            let mut x = 0.0;
            for j in 0..self.nrows(){
                x += self.buffer[i + j*self.row_size]
            }
            buffer.push(x)
        }
        Matrix {
            buffer,
            row_size: self.row_size,
            col_size: 1
        }
    }
    pub fn to_vec(self) -> Vec<f64>{
        assert!(self.col_size == 1);
        self.buffer
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