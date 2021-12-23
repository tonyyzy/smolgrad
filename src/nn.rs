use std::fmt::{Debug, Display};

use crate::engine::Value;
use rand::Rng;

trait Module {
    fn zero_grad(&self) {
        for v in self.parameters().iter_mut() {
            v.set_grad(0.0)
        }
    }

    fn parameters(&self) -> Vec<Value> {
        vec![]
    }
}

struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct MLP {
    sz: Vec<usize>,
    layers: Vec<Layer>,
}

impl Neuron {
    fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..=1.0)))
            .collect();
        Self {
            w,
            b: Value::new(0.0),
            nonlin,
        }
    }

    fn call(&self, x: &[Value]) -> Value {
        let act = self.w.iter().zip(x.iter()).fold(
            Value::new(0.0),
            |mut acc, (a, b)| {
                acc = acc + a * b;
                acc
            },
        ) + &self.b;
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut out = self.w.clone();
        out.push(self.b.clone());
        out
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = if self.nonlin { "ReLU" } else { "Linear" };
        f.write_fmt(format_args!("{} Neuron{}", ty, self.w.len()))
    }
}

impl Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
    }
}

impl Layer {
    fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin)).collect();
        Self { neurons }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Layer of {:?}", self.neurons))
    }
}

impl MLP {
    fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sz = vec![nin];
        sz.extend_from_slice(nouts);
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sz[i], sz[i + 1], i != (nouts.len() - 1)))
            .collect();
        Self { sz, layers }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        self.layers.iter().fold(x.to_vec(), |mut acc, layer| {
            acc = layer.call(&acc);
            acc
        })
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("MLP of {:?}", self.layers))
    }
}

mod test {
    use super::*;
    #[test]
    fn test_neuron() {
        let a = Neuron::new(10, true);
        // println!("{:?}", a.w);
        assert!(a.w.len() == 10);
        assert!(a.b.get_data() == 0.0);
        a.zero_grad();
        println!("{}", a);
        assert!(a.w[0].get_grad() == 0.0);
    }

    #[test]
    fn test_layer() {
        let a = Layer::new(8, 2, false);
        // println!("{:?}", a.w);
        assert!(a.neurons.len() == 2);
        assert!(a.neurons.first().unwrap().b.get_data() == 0.0);
        a.zero_grad();
        println!("{:?}", a);
        assert!(a.neurons.first().unwrap().w[0].get_grad() == 0.0);
    }

    #[test]
    fn test_MLP() {
        let a = MLP::new(8, &[4, 2]);
        assert!(a.sz.len() == 3);
        assert!(a.layers.first().unwrap().neurons.len() == 4);
        a.zero_grad();
        println!("{}", a);
        assert!(
            a.layers.first().unwrap().neurons.first().unwrap().w[0].get_grad()
                == 0.0
        );
    }
}
