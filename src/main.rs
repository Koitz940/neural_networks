mod network;
mod perceptrons;
mod sigmoid;

use network::Network;
fn main() {
    let net1 = Network::new(2, vec![2, 1]);
    let input = [1.0, 0.12];
    println!("{:#?}", net1);
    println!("{:#?}", net1.output(&input));
}
