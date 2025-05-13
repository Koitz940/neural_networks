mod matrices;
mod network;

use std::fs::File;
use std::io::Read;
use std::iter::repeat_with;

use crate::network::NeuralNetwork;
use matrices::Matrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::iter::zip;

fn main() {
    let mut net = NeuralNetwork::new(784, vec![30, 10], -1.0);
    fn train(epoch_amount: usize, batch_size: usize, net: &mut NeuralNetwork) {
        println!("starting learning...");
        let seed: [u8; 32] = rand::random(); // You can use any array of 32 bytes

        // Create two RNGs with the same seed
        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        let mut file = File::open("data\\train-images.idx3-ubyte").expect("Failed to open file");
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .expect("Failed to read file");
        let contents: Vec<f64> = contents.into_iter().map(|x| (x as f64) / 256.0).collect();
        let mut imgs: Vec<Vec<f64>> = contents[16..]
            .chunks_exact(784)
            .map(|i| i.to_vec())
            .collect();
        let mut file = File::open("data\\train-labels-idx1-ubyte").expect("Failed to open file");
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .expect("Failed to read file");
        let mut labels = Vec::new();
        for d in contents.iter().skip(8) {
            labels.push(vec_label(*d as usize))
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

    fn test(n: &NeuralNetwork) {
        println!("starting testing...");
        let mut count = 0.0;
        let mut total = 0.0;
        let mut file = File::open("data\\t10k-images-idx3-ubyte").expect("Failed to open file");
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .expect("Failed to read file");
        let contents: Vec<f64> = contents.into_iter().map(|x| (x as f64) / 256.0).collect();
        let imgs: Vec<Vec<f64>> = contents[16..]
            .chunks_exact(784)
            .map(|i| i.to_vec())
            .collect();
        let mut file = File::open("data\\t10k-labels-idx1-ubyte").expect("Failed to open file");
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .expect("Failed to read file");
        let mut labels = Vec::new();
        for d in contents.iter().skip(8) {
            labels.push(vec_label(*d as usize))
        }
        for (img, label) in zip(imgs, labels) {
            count += n.check_one(
                Matrix::into_matrix(vec![img]).unwrap(),
                Matrix::into_matrix(vec![label]).unwrap(),
            );
            total += 1.0
        }

        println!(
            "success rate: {count}/{total}  ({}%)",
            100.0 * count / total
        )
    }

    train(10000, 30, &mut net);
    test(&net)
}

fn vec_label(d: usize) -> Vec<f64> {
    let mut v: Vec<f64> = repeat_with(|| 0.0).take(10).collect();
    v[d] = 1.0;
    v
}
