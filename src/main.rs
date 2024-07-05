pub mod back_prop;
pub mod engine;
pub mod nn;
use engine::V;

fn main() {
    let a = v!(2.0);
    let b = v!(4.0);
    let c = v!(5.0);
    let d = v!(1.0);
    let f = v!(4.0);
    let g = v!(8.0);

    let e = ((f * (d * (c + b))) / a) / g;

    let mut e = e.relu();
    e.backward();

    for d in e.build_topo(&mut vec![]) {
        println!("data = {}; grad = {}; op = {:?}", d.data, d.grad, d.op)
    }
}
