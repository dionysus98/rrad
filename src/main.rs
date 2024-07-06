pub mod back_prop;
pub mod engine;
pub mod nn;
pub mod rand;
use engine::V;

fn main() {
    let a = v!(2.0, "a");
    let b = v!(4.0, "b");
    let c = v!(5.0, "c");
    let d = v!(1.0, "d");
    let f = v!(4.0, "f");
    let g = v!(8.0);

    let one = &mut (f * d);
    let two = &mut (c + b);
    let three = &mut (one * two);
    let e = &mut (three / g);
    let e = &mut (e + a);

    let mut e = e.relu();
    e.backward();

    for d in e.build_topo(&mut vec![]) {
        println!(
            "data = {}; grad = {}; op = {:?}; label = {:?}",
            d.data, d.grad, d.op, d.label
        )
    }

    // let n = n!(3);

    // dbg!(n)
}
