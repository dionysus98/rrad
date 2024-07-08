use std::iter::{zip, Sum};

use crate::engine::V;
use crate::nn::MLP;
use crate::v;
// # loss function
// def loss(batch_size=None):

//     # inline DataLoader :)
//     if batch_size is None:
//         Xb, yb = X, y
//     else:
//         ri = np.random.permutation(X.shape[0])[:batch_size]
//         Xb, yb = X[ri], y[ri]
//     inputs = [list(map(Value, xrow)) for xrow in Xb]

//     # forward the model to get scores
//     scores = list(map(model, inputs))

//     # svm "max-margin" loss
//     losses = [(1 + -yi * scorei).relu()
//               for yi, scorei in zip(yb, scores)]
//     data_loss = sum(losses) * (1.0 / len(losses))
//     # L2 regularization
//     alpha = 1e-4
//     reg_loss = alpha * sum((p * p for p in model.parameters()))
//     total_loss = data_loss + reg_loss

//     # also get accuracy
//     accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
//     return total_loss, sum(accuracy) / len(accuracy)

// total_loss, acc = loss()
// print(total_loss, acc)

pub fn loss(model: &MLP, mut xb: Vec<Vec<f32>>, yb: Vec<f32>) {
    let scores: Vec<V> = xb.iter_mut().map(|xrow| model.call(xrow)).last().unwrap();
    let losses: Vec<V> = zip(yb, scores)
        .map(|(yi, si)| (v!(1.0) + v!(-yi) * si).relu())
        .collect();

    let loss_len = losses.len();

    let data_loss = losses.into_iter().sum::<V>() * v!(1.0 / loss_len as f32);
    // 
}
