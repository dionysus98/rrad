pub mod back_prop;
pub mod engine;
// pub mod loss;
pub mod nn;
pub mod rand;

use std::iter::zip;

use engine::V;
use nn::Module;

fn main() {
    // let a = v!(2.0, "a");
    // let b = v!(4.0, "b");
    // let c = v!(5.0, "c");
    // let d = v!(1.0, "d");
    // let f = v!(4.0, "f");
    // let g = v!(8.0);

    // let one = f * d;
    // let two = c + b;
    // let three = one * two;
    // let e = three / g;
    // let e = e + a;

    // let mut e = e.tanh().relu().relu().tanh();
    // e.backward();

    // for d in e.build_topo(&mut vec![]) {
    //     println!(
    //         "data = {}; grad = {}; op = {:?}; label = {:?}",
    //         d.data, d.grad, d.op, d.label
    //     )
    // }

    let mut n = MLP!(3, &[4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let ys = vec![1.0, -1.0, -1.0, 1.0];

    // println!("{:?}", n.layers[0].neurons[0].w);

    // let mut a  = &n.parameters();

    // n.parameters().iter_mut().next().unwrap().data = 8.0;

    // dbg!(n.parameters());

    // for k in 0..5 {
    //     // forward pass
    //     let ypred = xs.iter().map(|xrow| n.call(xrow)).last().unwrap();
    //     let mut loss = zip(ys.clone(), ypred)
    //         .map(|(ygt, yout)| (yout - v!(ygt)).powf(2.0))
    //         .sum::<V>();

    //     // backward pass
    //     n.zero_grad();
    //     loss.backward();

    //     // update
    //     for p in n.parameters() {
    //         p.data += -0.1 * p.grad
    //     }

    //     let loss_topo = loss.build_topo(&mut vec![]);

    //     println!("{}", loss_topo.len());

    //     println!("{:?}", n.layers[0].neurons[0].w.len());

    //     // for d in loss_topo {
    //     //     println!("op = {:?}; label = {:?}", d.op, d.label)
    //     // }

    //     // println!("k = {:?}, loss = {:?}", k, loss.data)
    // }

    // println!("{:?}", n.layers[0].neurons[0].w);
}
