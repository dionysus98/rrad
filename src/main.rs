pub mod back_prop;
pub mod engine;
pub mod nn;
use engine::V;

fn main() {
    let a = V::new(2.0);
    let b = V::new(4.0);
    let c = V::new(5.0);
    let d = V::new(1.0);
    let f = V::new(4.0);
    let g = V::new(8.0);

    let e = ((f * (d * (c + b))) / a) / g;

    let e = e.relu();

    for d in e.backward() {
        println!("data = {}; grad = {}; op = {:?}", d.data, d.grad, d.op)
    }
}
