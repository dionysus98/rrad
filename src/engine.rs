use crate::back_prop;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub enum Vops {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Relu,
    Tanh,
    Sigm,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct V {
    pub data: f32,
    pub grad: f32,
    pub op: Vops,
    pub children: Vec<V>,
    pub context: Option<f32>,
}

impl V {
    pub fn new(data: f32) -> Self {
        Self {
            data,
            grad: 0.0,
            op: Vops::None,
            children: vec![],
            context: None,
        }
    }

    pub fn powf(self, pow: f32) -> Self {
        Self {
            data: self.data.powf(pow),
            grad: self.grad,
            op: Vops::Exp,
            children: vec![self],
            context: Some(pow),
        }
    }

    pub fn relu(self) -> Self {
        Self {
            data: if self.data < 0.0 { 0.0 } else { self.data },
            grad: self.grad,
            op: Vops::Relu,
            children: vec![self],
            context: None,
        }
    }

    pub fn build_topo(&mut self, visited: &mut Vec<Self>) -> Vec<Self> {
        if !visited.contains(self) {
            visited.push(self.clone());
            for cl in self.children.iter_mut() {
                cl.build_topo(visited);
            }
            return visited.to_vec();
        }
        visited.to_vec()
    }

    fn backward_recur(&mut self) {
        back_prop::back_propogate(self);
        if !self.children.is_empty() {
            for cl in self.children.iter_mut() {
                cl.backward_recur()
            }
        }
    }

    pub fn backward(&mut self) {
        self.grad = 1.0;
        self.backward_recur();
    }
}

impl Add for V {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            data: self.data + other.data,
            grad: self.grad,
            op: Vops::Add,
            children: vec![self, other],
            context: None,
        }
    }
}

impl Sub for V {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            data: self.data - other.data,
            grad: self.grad,
            op: Vops::Sub,
            children: vec![self, other],
            context: None,
        }
    }
}

impl Mul for V {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            data: self.data * other.data,
            grad: self.grad,
            op: Vops::Mul,
            children: vec![self, other],
            context: None,
        }
    }
}

impl Div for V {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self {
            data: self.data / other.data,
            grad: self.grad,
            op: Vops::Div,
            children: vec![self, other],
            context: None,
        }
    }
}

#[macro_export]
macro_rules! v {
    ( $x:expr ) => {
        V::new($x)
    };
}
