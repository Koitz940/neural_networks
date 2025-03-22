mod mat_op;
mod matrices;
mod network;

use std::fs::File;
use std::io::{self, BufRead};
use std::iter::repeat_with;
use std::path::Path;

use crate::network::NeuralNetwork;
use matrices::Matrix;
fn main() {
    let mut net = NeuralNetwork::new(784, vec![16, 16, 10], -10.0);
    fn train(m: usize, n: &mut NeuralNetwork) {
        println!("starting learning...");
        if let Ok(lines) = read_lines("mnist_train.txt") {
            let mut minibatch: Vec<Vec<f64>> = Vec::new();
            let mut labels: Vec<Vec<f64>> = Vec::new();
            let mut i = 0;
            for line in lines.map_while(Result::ok) {
                let mut digits = line.split(",");
                let label = vec_label(digits.next().unwrap().parse().unwrap());
                let img = digits
                    .map(|d| d.parse::<f64>().unwrap() / 256.0)
                    .collect::<Vec<f64>>();
                minibatch.push(img);
                labels.push(label);

                i += 1;
                if i == m {
                    n.learn(
                        &Matrix::into_matrix(minibatch.clone()).unwrap(),
                        &Matrix::into_matrix(labels.clone()).unwrap(),
                    );
                    minibatch = Vec::new();
                    labels = Vec::new();
                    i = 0;
                }
            }
            if i != 0 {
                n.learn(
                    &Matrix::into_matrix(minibatch.clone()).unwrap(),
                    &Matrix::into_matrix(labels.clone()).unwrap(),
                );
            }
            println!("reading completed")
        }
    }

    fn test(n: &NeuralNetwork) {
        println!("starting testing");
        let mut count = 0.0;
        let mut total = 0.0;
        if let Ok(lines) = read_lines("mnist_test.txt") {
            for line in lines.map_while(Result::ok) {
                let mut digits = line.split(",");
                let label =
                    Matrix::into_matrix(vec![vec_label(digits.next().unwrap().parse().unwrap())])
                        .unwrap();
                let img = Matrix::into_matrix(vec![digits
                    .map(|x| (x.parse::<f64>().unwrap()) / 256.0)
                    .collect()])
                .unwrap();
                count += n.check_one(img, label);
                total += 1.0
            }
        }

        println!(
            "success rate: {count}/{total}  ({}%)",
            100.0 * count / total
        )
    }

    train(100, &mut net);
    test(&net)
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn vec_label(d: usize) -> Vec<f64> {
    let mut v: Vec<f64> = repeat_with(|| 0.0).take(10).collect();
    v[d] = 1.0;
    v
}
