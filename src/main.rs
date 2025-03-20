mod mat_op;
mod matrices;
mod network;
mod networks_2;
mod perceptrons;
mod sigmoid;

use std::fs::File;
use std::io::{self, BufRead};
use std::iter::repeat_with;
use std::path::Path;

use matrices::Matrix;
use networks_2::NeuralNetwork;
fn main() {
    let mut net = networks_2::NeuralNetwork::new(784, vec![16, 16, 10]);
    fn train(m: usize, n: &mut NeuralNetwork) {
        let mut training = Vec::new();
        println!("Starting training...");
        if let Ok(lines) = read_lines("mnist_train.txt") {
            for line in lines.map_while(Result::ok) {
                let mut digits = line.split(",");
                let label: usize = digits.next().unwrap().parse().unwrap();
                let img = digits
                    .map(|d| vec![d.parse::<f64>().unwrap() / 256.0])
                    .collect::<Vec<Vec<f64>>>();
                let label = vec_label(label);
                training.push(vec![img, label]);
            }
        }
        //to be finished
    }
    let mut count = 0.0;
    let mut total = 0.0;
    println!("finished training, starting testing...");
    println!(
        "success rate: {count}/{total}  ({}%)",
        100.0 * count / total
    )
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn train_vec_label() {}

fn vec_label(d: usize) -> matrices::Matrix {
    let mut v: Vec<Vec<f64>> = repeat_with(|| vec![0.0]).take(10).collect();
    v[d][0] = 1.0;
    Matrix::into_matrix(v).unwrap()
}
