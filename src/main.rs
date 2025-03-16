mod mat_op;
mod network;
mod networks_2;
mod perceptrons;
mod sigmoid;

use std::fs::File;
use std::io::{self, BufRead};
use std::iter::repeat_with;
use std::path::Path;
fn main() {
    let mut net = networks_2::NeuralNetwork::new(784, vec![10]);
    println!("Starting training...");
    if let Ok(lines) = read_lines("mnist_train.txt") {
        for line in lines.map_while(Result::ok) {
            let mut digits = line.split(",");
            let label: usize = digits.next().unwrap().parse().unwrap();
            let img: Box<[f64]> = digits.map(|d| d.parse::<f64>().unwrap() / 256.0).collect();
            let label = vec_label(label);
            net.update_one(&img, &label);
            drop(img);
        }
    }
    let mut count = 0.0;
    let mut total = 0.0;
    println!("finished training, starting testing...");
    if let Ok(lines) = read_lines("mnist_test.txt") {
        for line in lines.map_while(Result::ok) {
            let mut digits = line.split(",");
            let label: usize = digits.next().unwrap().parse().unwrap();
            let img: Box<[f64]> = digits.map(|d| d.parse::<f64>().unwrap() / 256.0).collect();

            count += net.test_one(&img, &vec_label(label));
            total += 1.0;
            drop(img);
        }
    }
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

fn vec_label(d: usize) -> Box<[f64]> {
    let mut v: Box<[f64]> = repeat_with(|| 0.0).take(10).collect();
    v[d] = 1.0;
    v
}
