use crate::back_prop;
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Sub},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Vops {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    Tanh,
    Sigm,
    None,
}

#[derive(Debug, PartialEq)]
pub struct V<'a> {
    pub data: f32,
    pub grad: f32,
    pub op: Vops,
    pub children: Vec<&'a mut V<'a>>,
    pub context: Option<f32>,
    pub label: Option<&'static str>,
}

// just to build the topology
impl<'a> Clone for V<'a> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            grad: self.grad.clone(),
            op: self.op.clone(),
            children: vec![],
            context: self.context.clone(),
            label: self.label.clone(),
        }
    }
}

impl<'a> V<'a> {
    pub fn new(data: f32, label: Option<&'static str>) -> Self {
        Self {
            data,
            grad: 0.0,
            op: Vops::None,
            children: vec![],
            context: None,
            label,
        }
    }

    pub fn powf(&'a mut self, pow: f32) -> Self {
        Self {
            data: self.data.powf(pow),
            grad: self.grad,
            op: Vops::Pow,
            children: vec![self],
            context: Some(pow),
            label: None,
        }
    }

    pub fn relu(&'a mut self) -> Self {
        Self {
            data: if self.data < 0.0 { 0.0 } else { self.data },
            grad: 0.0,
            op: Vops::Relu,
            children: vec![self],
            context: None,
            label: None,
        }
    }

    pub fn tanh(&'a mut self) -> Self {
        let x = self.data;
        let xe = 2.0 * x;
        let t = (xe - 1.0) / xe + 1.0;
        Self {
            data: t,
            grad: 0.0,
            op: Vops::Tanh,
            children: vec![self],
            context: Some(t),
            label: None,
        }
    }

    pub fn build_topo(&mut self, visited: &mut Vec<Self>) -> Vec<Self> {
        if !visited.contains(&self) {
            visited.push(self.clone());
            for cl in self.children.iter_mut() {
                cl.build_topo(visited);
            }
        }
        visited.to_vec()
    }

    fn backward_recur(&'a mut self) {
        back_prop::back_propogate(self);
        if !self.children.is_empty() {
            for cl in self.children.iter_mut() {
                cl.backward_recur()
            }
        }
    }

    pub fn backward(&'a mut self) {
        self.grad = 1.0;
        self.backward_recur();
    }
}

// impl<'a> Sum for &'a mut V<'a> {
//     fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
//         iter.fold(V::new(0.0, None), |a, b| (a + b))
//     }
// }

impl<'a> Add for &'a mut V<'a> {
    type Output = V<'a>;

    fn add(self, other: Self) -> Self::Output {
        V {
            data: self.data + other.data,
            grad: self.grad,
            op: Vops::Add,
            children: vec![self, other],
            context: None,
            label: None,
        }
    }
}

impl<'a> Sub for &'a mut V<'a> {
    type Output = V<'a>;

    fn sub(self, other: Self) -> Self::Output {
        V {
            data: self.data + other.data,
            grad: self.grad,
            op: Vops::Sub,
            children: vec![self, other],
            context: None,
            label: None,
        }
    }
}

impl<'a> Mul for &'a mut V<'a> {
    type Output = V<'a>;

    fn mul(self, other: Self) -> Self::Output {
        V {
            data: self.data * other.data,
            grad: self.grad,
            op: Vops::Mul,
            children: vec![self, other],
            context: None,
            label: None,
        }
    }
}

impl<'a> Div for &'a mut V<'a> {
    type Output = V<'a>;

    fn div(self, other: Self) -> Self::Output {
        V {
            data: self.data / other.data,
            grad: self.grad,
            op: Vops::Div,
            children: vec![self, other],
            context: None,
            label: None,
        }
    }
}

#[macro_export]
macro_rules! v {
    ( $x:expr ) => {{
        crate::engine::V::new($x, None)
    }};

    ( $x:expr, $y:expr ) => {
        crate::engine::V::new($x, Some($y))
    };
}
