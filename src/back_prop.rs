use crate::engine::V;

pub fn add(mut v: V) -> V {
    for cl in v.children.iter_mut() {
        cl.grad += v.grad;
    }
    v
}

pub fn sub(mut v: V) -> V {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += v.grad;
            cl_b.grad -= v.grad;
        }
    };
    v
}

pub fn mul(mut v: V) -> V {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += cl_b.data * v.grad;
            cl_b.grad += cl_a.data * v.grad;
        }
    };
    v
}

pub fn div(mut v: V) -> V {
    let mut cls = v.children.iter_mut();
    if let Some(cl_a) = cls.next() {
        if let Some(cl_b) = cls.next() {
            cl_a.grad += (cl_b.data.powf(-1.0)) * v.grad;
            cl_b.grad += (-cl_a.data / cl_b.data.powf(2.0)) * v.grad;
        }
    };
    v
}

pub fn exp(mut v: V) -> V {
    if let Some(cl) = v.children.iter_mut().next() {
        cl.grad += v.grad;
    }
    v
}

pub fn relu(mut v: V) -> V {
    if let Some(cl) = v.children.iter_mut().next() {
        let data = if v.data > 0.0 { 1.0 } else { 0.0 };
        cl.grad += v.grad * data;
    }
    v
}
