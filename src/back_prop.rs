use crate::engine::{Vops, V};

pub fn add(v: &mut V) {
    for cl in v.children.iter_mut() {
        cl.grad += v.grad;
    }
}

pub fn sub(v: &mut V) {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += v.grad;
            cl_b.grad -= v.grad;
        }
    };
}

pub fn mul(v: &mut V) {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += cl_b.data * v.grad;
            cl_b.grad += cl_a.data * v.grad;
        }
    };
}

pub fn div(v: &mut V) {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += (cl_b.data.powf(-1.0)) * v.grad;
            // derivative for div -1 / x**2
            cl_b.grad += (-cl_a.data / cl_b.data.powf(2.0)) * v.grad;
        }
    };
}

pub fn pow(v: &mut V) {
    if let Some(cl) = v.children.iter_mut().next() {
        if let Some(pow) = v.context {
            cl.grad += pow * (v.data.powf(pow - 1.0)) * v.grad;
        }
    }
}

pub fn relu(v: &mut V) {
    if let Some(cl) = v.children.iter_mut().next() {
        let data = if v.data > 0.0 { 1.0 } else { 0.0 };
        cl.grad += v.grad * data;
    }
}

pub fn tanh(v: &mut V) {
    if let Some(cl) = v.children.iter_mut().next() {
        if let Some(t) = v.context {
            // (1 - t**2) * out.grad
            cl.grad += (1.0 - t.powf(2.0)) * v.grad;
        }
    }
}

pub fn back_propogate(v: &mut V) {
    match v.op {
        Vops::Add => add(v),
        Vops::Sub => sub(v),
        Vops::Mul => mul(v),
        Vops::Div => div(v),
        Vops::Pow => pow(v),
        Vops::Relu => relu(v),
        Vops::Tanh => tanh(v),
        Vops::Sigm => todo!(),
        Vops::None => (),
    }
}
