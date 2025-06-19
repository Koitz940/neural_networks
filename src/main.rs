mod matrices;
mod network;

use std::fs::File;
use std::io::Read;
use std::iter::repeat_with;
use std::path::PathBuf;

use crate::network::NeuralNetwork;
use matrices::Matrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::iter::zip;

fn main() {
    let mut net = NeuralNetwork::new(784, vec![100, 10], 1.5, 0.01);
    fn train(epoch_amount: usize, batch_size: usize, net: &mut NeuralNetwork) {
        println!("starting learning...");
        let seed: [u8; 32] = rand::random(); // You can use any array of 32 bytes

        // Create two RNGs with the same seed
        let mut rng1 = StdRng::from_seed(seed);
        let mut rng2 = StdRng::from_seed(seed);

        let train_img_path = ["data", "train-images.idx3-ubyte"]
            .iter()
            .collect::<PathBuf>();
        let mut file = File::open(train_img_path).expect("Failed to open file");

        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .expect("Failed to read file");

        let contents: Vec<f64> = contents.into_iter().map(|x| (x as f64) / 256.0).collect();

        let mut imgs: Vec<Vec<f64>> = contents[16..]
            .chunks_exact(784)
            .map(|i| i.to_vec())
            .collect();

        let train_lbl_path = ["data", "train-labels.idx1-ubyte"]
            .iter()
            .collect::<PathBuf>();
        let mut file = File::open(train_lbl_path).expect("Failed to open file");

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

            let mut sample_imgs = Vec::with_capacity(784 * batch_size);
            for img in 0..batch_size {
                for i in 0..784 {
                    sample_imgs.push(imgs[img][i])
                }
            }

            let mut sample_labels = Vec::with_capacity(10 * batch_size);
            for label in 0..batch_size {
                for i in 0..10 {
                    sample_labels.push(labels[label][i])
                }
            }

            net.learn(
                Matrix::into_matrix(sample_imgs, 784, batch_size).unwrap(),
                Matrix::into_matrix(sample_labels, 10, batch_size).unwrap(),
            );
        }
        println!("reading completed")
    }

    let test_img_path = ["data", "t10k-images.idx3-ubyte"]
        .iter()
        .collect::<PathBuf>();
    let mut file = File::open(test_img_path).expect("Failed to open file");

    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .expect("Failed to read file");

    let contents: Vec<f64> = contents.into_iter().map(|x| (x as f64) / 256.0).collect();

    let imgs: Vec<Vec<f64>> = contents[16..]
        .chunks_exact(784)
        .map(|i| i.to_vec())
        .collect();

    let test_lbl_path = ["data", "t10k-labels.idx1-ubyte"]
        .iter()
        .collect::<PathBuf>();
    let mut file = File::open(test_lbl_path).expect("Failed to open file");

    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .expect("Failed to read file");

    let mut labels = Vec::new();
    for d in contents.iter().skip(8) {
        labels.push(vec_label(*d as usize))
    }

    fn test(n: &NeuralNetwork, images: Vec<Vec<f64>>, labs: Vec<Vec<f64>>) -> (u32, u32, f64) {
        let mut count = 0.0;
        let mut total = 0.0;

        for (img, label) in zip(images, labs) {
            count += n.check_one(
                Matrix::into_matrix(img, 784, 1).unwrap(),
                Matrix::into_matrix(label, 10, 1).unwrap(),
            );
            total += 1.0
        }

        (count as u32, total as u32, 100.0 * count / total)
    }

    train(2000, 100, &mut net);

    println!("starting testing...");

    let results = test(&net, imgs, labels);
    println!("success rate: {}/{}; {}%", results.0, results.1, results.2)
}

fn vec_label(d: usize) -> Vec<f64> {
    let mut v: Vec<f64> = repeat_with(|| 0.0).take(10).collect();
    v[d] = 1.0;
    v
}
