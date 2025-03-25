mod matrices;
mod network;

use std::fs::File;
use std::io::{self, BufRead};
use std::iter::repeat_with;
use std::path::Path;

use crate::network::NeuralNetwork;
use matrices::Matrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
fn main() {
    let mut net = NeuralNetwork::new(784, vec![30, 10], -0.5);
    fn train(epoch_amount: usize, batch_size: usize, net: &mut NeuralNetwork) {
        println!("starting learning...");
        let seed: [u8; 32] = rand::random(); // You can use any array of 32 bytes

        // Create two RNGs with the same seed
        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        if let Ok(lines) = read_lines("mnist_train.txt") {
            let mut imgs = Vec::new();
            let mut labels = Vec::new();
            for line in lines.map_while(Result::ok) {
                let mut digits = line.split(",");
                let label = vec_label(digits.next().unwrap().parse().unwrap());
                let img = digits
                    .map(|d| d.parse::<f64>().unwrap() / 256.0)
                    .collect::<Vec<f64>>();
                imgs.push(img);
                labels.push(label);
            }
            for _ in 0..epoch_amount {
                imgs.shuffle(&mut rng1);
                labels.shuffle(&mut rng2);
                let sample_imgs = &imgs[..batch_size];
                let sample_labels = &labels[..batch_size];
                net.learn(
                    &Matrix::new(sample_imgs).unwrap(),
                    &Matrix::new(sample_labels).unwrap(),
                );
            }
            println!("reading completed")
        }
    }

    fn test(n: &NeuralNetwork) {
        println!("starting testing...");
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

    train(100000, 30, &mut net);
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
