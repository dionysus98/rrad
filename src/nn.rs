use std::iter::zip;

use crate::engine::V;
use crate::rand::rand;
use crate::v;

pub trait Module<'a> {
    fn zero_grad(&'a mut self) {
        for p in self.parameters().iter_mut() {
            p.grad = 0.0
        }
    }

    fn parameters(&'a mut self) -> Vec<&'a mut V<'a>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Neuron<'a> {
    pub w: Vec<V<'a>>,
    pub b: V<'a>,
    pub nonlin: bool,
}

impl<'a> Neuron<'a> {
    pub fn new(nin: i32, nonlin: bool) -> Self {
        Self {
            w: (0..nin).map(|_| v!(rand())).collect(),
            b: v!(rand()),
            nonlin,
        }
    }

    // pub fn call(&'a mut self, xs: &'a mut Vec<V<'a>>) -> V {
    //     let mut dotp = zip(&mut self.w, xs)
    //         .map(|(wi, xi)| (wi * xi))
    //         .collect::<Vec<V>>();

    //     // let a =

    //     // if self.nonlin {
    //     //     act.tanh()
    //     // } else {
    //     //     act
    //     // }
    //     todo!()
    // }
}

impl<'a> Module<'a> for Neuron<'a> {
    fn parameters(&'a mut self) -> Vec<&'a mut V<'a>> {
        let mut ps: Vec<&'a mut V> = self.w.iter_mut().map(|v| v).collect();
        ps.push(&mut self.b);
        ps
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer<'a> {
    pub neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    pub fn new(nin: i32, nout: i32, nonlin: bool) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    // pub fn call(&mut self, xs: &'a mut Vec<V<'a>>) -> V {
    //     self.neurons.iter_mut().map(|n| n.call(xs)).last().unwrap()
    // }
}

impl<'a> Module<'a> for Layer<'a> {
    fn parameters(&'a mut self) -> Vec<&'a mut V<'a>> {
        self.neurons
            .iter_mut()
            .flat_map(|v| v.parameters())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MLP<'a> {
    pub layers: Vec<Layer<'a>>,
}

impl<'a> MLP<'a> {
    pub fn new(nin: i32, nouts: &[i32]) -> Self {
        let mut nin = vec![nin];
        let total = nouts.len();
        nin.append(&mut nouts.to_vec());

        let nin = &nin;

        let layers = (0..total)
            .map(|i| Layer::new(nin[i], nin[i + 1], i == total))
            .collect();

        Self { layers }
    }

    // pub fn call(&'a mut self, xs: &'a mut Vec<V<'a>>) -> Vec<V> {
    //     self.layers.iter_mut().map(|l| l.call(xs)).collect()
    // }
}

impl<'a> Module<'a> for MLP<'a> {
    fn parameters(&'a mut self) -> Vec<&'a mut V<'a>> {
        self.layers
            .iter_mut()
            .flat_map(|v| v.parameters())
            .collect()
    }
}

#[macro_export]
macro_rules! n {
    ( $x:expr ) => {
        crate::nn::Neuron::new($x, true)
    };
}

#[macro_export]
macro_rules! MLP {
    ( $nin:expr, $nouts:expr ) => {
        crate::nn::MLP::new($nin, $nouts)
    };
}
