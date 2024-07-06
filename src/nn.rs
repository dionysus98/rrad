// use std::iter::{zip, Sum};

// use crate::engine::V;
// use crate::rand::rand;
// use crate::v;

// pub trait Module {
//     fn zero_grad(&mut self) {
//         for p in self.parameters().iter_mut() {
//             p.grad = 0.0
//         }
//     }

//     fn parameters(&mut self) -> Vec<&mut V>;
// }

// pub struct Neuron<'a> {
//     pub w: Vec<V<'a>>,
//     pub b: V<'a>,
//     pub nonlin: bool,
// }

// impl<'a> Neuron<'a> {
//     pub fn new(nin: i32, nonlin: bool) -> Self {
//         Self {
//             w: (0..nin).map(|_| v!(rand())).collect(),
//             b: v!(rand()),
//             nonlin,
//         }
//     }

//     // pub fn call(&mut self, xs: Vec<&'a mut V<'a>>) -> V<'a> {
//     //     let mut act = &mut self.b;

//     //     for (mut wi, xi) in zip(self.w.clone(), xs) {
//     //         let mut mul = &mut wi * xi;
//     //         let op = act + &mut mul;
//     //         act = op;
//     //     }

//     //     let act = act;

//     //     if self.nonlin {
//     //         act.clone()
//     //     } else {
//     //         act.clone()
//     //     }
//     // }
// }

// impl<'a> Module for Neuron<'a> {
//     fn parameters(&mut self) -> Vec<&mut V> {
//         let mut ps: Vec<&mut V> = self.w.iter_mut().map(|v| v).collect();
//         ps.push(&mut self.b);
//         ps
//     }
// }

// struct Layer<'a> {
//     neurons: Vec<Neuron<'a>>,
// }

// impl<'a> Module for Layer<'a> {
//     fn parameters(&mut self) -> Vec<&mut V> {
//         self.neurons
//             .iter_mut()
//             .flat_map(|v| v.parameters())
//             .collect()
//     }
// }

// struct MLP<'a> {
//     layers: Vec<Layer<'a>>,
// }

// impl<'a> Module for MLP<'a> {
//     fn parameters(&mut self) -> Vec<&mut V> {
//         self.layers
//             .iter_mut()
//             .flat_map(|v| v.parameters())
//             .collect()
//     }
// }

// #[macro_export]
// macro_rules! n {
//     ( $x:expr ) => {
//         Neuron::new($x, true)
//     };
// }
