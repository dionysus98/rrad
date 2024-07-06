use std::iter::zip;

use crate::engine::V;
use crate::rand::rand;
use crate::v;

pub trait Module {
    fn zero_grad(&mut self) {
        for p in self.parameters().iter_mut() {
            p.grad = 0.0
        }
    }

    fn parameters(&mut self) -> Vec<&mut V>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    pub w: Vec<V>,
    pub b: V,
    pub nonlin: bool,
}

impl Neuron {
    pub fn new(nin: i32, nonlin: bool) -> Self {
        Self {
            w: (0..nin).map(|_| v!(rand())).collect(),
            b: v!(rand()),
            nonlin,
        }
    }

    pub fn call(&self, xs: &Vec<f32>) -> V {
        let mut act = self.b.clone();
        for (wi, xi) in zip(self.w.clone(), xs) {
            act = act + wi * v!(*xi);
        }

        if self.nonlin {
            act
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&mut self) -> Vec<&mut V> {
        let mut ps: Vec<&mut V> = self.w.iter_mut().map(|v| v).collect();
        ps.push(&mut self.b);
        ps
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: i32, nout: i32, nonlin: bool) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    pub fn call(&self, xs: &Vec<f32>) -> Vec<V> {
        self.neurons.iter().map(|n| n.call(&xs)).collect()
    }
}

impl Module for Layer {
    fn parameters(&mut self) -> Vec<&mut V> {
        self.neurons
            .iter_mut()
            .flat_map(|v| v.parameters())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
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

    pub fn call(&self, xs: &Vec<f32>) -> Vec<V> {
        self.layers.iter().flat_map(|l| l.call(&xs)).collect()
    }
}

impl Module for MLP {
    fn parameters(&mut self) -> Vec<&mut V> {
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
