pub mod back_prop;
pub mod engine;
use engine::V;

fn main() {
    let v1 = V::new(3.0);
    let v2 = V::new(2.0);

    let v3 = v1 * v2;

    let v4 = v3 * V::new(5.0);

    for mut d in v4.relu().backward() {
        d.children = vec![];
        dbg!("{:?}", d);
    }
}
