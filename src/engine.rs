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
            data: data,
            grad: 0.0,
            op: Vops::None,
            children: vec![],
            context: None,
        }
    }

    pub fn powf(self, pow: f32) -> Self {
        let out = Self {
            data: self.data.powf(pow),
            grad: self.grad,
            op: Vops::Exp,
            children: vec![self],
            context: Some(pow),
        };

        out
    }

    pub fn relu(self) -> Self {
        let data = if self.data < 0.0 { 0.0 } else { self.data };
        let out = Self {
            data: data,
            grad: self.grad,
            op: Vops::Relu,
            children: vec![self],
            context: None,
        };
        out
    }

    fn _back_propogate(self) -> Self {
        match self.op {
            Vops::Add => back_prop::add(self),
            Vops::Sub => back_prop::sub(self),
            Vops::Mul => back_prop::mul(self),
            Vops::Div => back_prop::div(self),
            Vops::Exp => back_prop::exp(self),
            Vops::Relu => back_prop::relu(self),
            Vops::Tanh => todo!(),
            Vops::Sigm => todo!(),
            Vops::None => self,
        }
    }

    fn build_topo(&mut self, visited: &mut Vec<Self>) -> Vec<Self> {
        if !visited.contains(&self) {
            visited.push(self.clone());
            for cl in self.children.iter_mut() {
                cl.build_topo(visited);
            }
            return visited.to_vec();
        }
        visited.to_vec()
    }

    fn backward_recur(&mut self) -> Self {
        let mut v = self.clone()._back_propogate();
        let mut cls = vec![];
        if !self.children.is_empty() {
            for cl in v.children.iter_mut() {
                cls.push(cl.backward_recur())
            }
        }
        v.children = cls;
        v
    }

    pub fn backward(mut self) -> Vec<Self> {
        self.grad = 1.0;
        self.backward_recur().build_topo(&mut vec![])
    }
}

impl Add for V {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let out = Self {
            data: self.data + other.data,
            grad: self.grad,
            op: Vops::Add,
            children: vec![self, other],
            context: None,
        };
        out
    }
}

impl Sub for V {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let out = Self {
            data: self.data - other.data,
            grad: self.grad,
            op: Vops::Sub,
            children: vec![self, other],
            context: None,
        };
        out
    }
}

impl Mul for V {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let out = Self {
            data: self.data * other.data,
            grad: self.grad,
            op: Vops::Mul,
            children: vec![self, other],
            context: None,
        };
        out
    }
}

impl Div for V {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        let out = Self {
            data: self.data / other.data,
            grad: self.grad,
            op: Vops::Div,
            children: vec![self, other],
            context: None,
        };
        out
    }
}

#[macro_export]
macro_rules! v {
    ( $x:expr ) => {
        V::new($x)
    };
}
