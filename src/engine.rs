use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

enum Ops {
    Add,
    Mul,
    Pow,
    ReLU,
    None,
}

struct Inner {
    pub data: Rc<Cell<f32>>,
    pub grad: Rc<Cell<f32>>,
    backward: Box<dyn Fn() -> ()>,
    pub prev: Vec<Value>,
    op: Ops,
}

impl Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("prev", &self.prev)
            .finish()
    }
}

impl Display for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Value(Rc<RefCell<Inner>>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.get_data())
            .field("grad", &self.get_grad())
            .finish()
    }
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self(Rc::new(RefCell::new(Inner {
            data: Rc::new(Cell::new(data)),
            grad: Rc::new(Cell::new(0.0)),
            backward: Box::new(|| {}),
            prev: vec![],
            op: Ops::None,
        })))
    }

    fn _new(data: f32, prev: Vec<Self>, op: Ops) -> Self {
        Self(Rc::new(RefCell::new(Inner {
            data: Rc::new(Cell::new(data)),
            grad: Rc::new(Cell::new(0.0)),
            backward: Box::new(|| {}),
            prev,
            op,
        })))
    }

    fn backward(&self) {
        let mut topo = vec![];
        let mut visited = HashSet::new();
        fn build_topo(
            v: &Value,
            visited: &mut HashSet<Value>,
            topo: &mut Vec<Value>,
        ) {
            if !visited.contains(&v) {
                visited.insert(v.clone());
                for child in v.0.borrow().prev.iter() {
                    build_topo(child, visited, topo)
                }
                topo.push(v.clone())
            }
        }
        build_topo(self, &mut visited, &mut topo);
        self.0.borrow_mut().grad.set(1.0);
        for v in topo.iter().rev() {
            v.0.borrow_mut().backward.as_ref()();
            // println!("{:?} {}", v.0.as_ptr(), v);
        }
    }

    pub fn get_grad(&self) -> f32 {
        self.0.borrow().grad.get()
    }

    pub fn get_data(&self) -> f32 {
        self.0.borrow().data.get()
    }

    fn clone_grad(&self) -> Rc<Cell<f32>> {
        self.0.borrow().grad.clone()
    }
    fn clone_data(&self) -> Rc<Cell<f32>> {
        self.0.borrow().data.clone()
    }

    fn set_backward(&self, func: Box<dyn Fn() -> ()>) {
        self.0.borrow_mut().backward = func
    }

    pub fn set_grad(&self, grad: f32) {
        self.0.borrow().grad.set(grad)
    }

    fn pow(&self, rhs: f32) -> Self {
        let out = Value::_new(
            self.get_data().powf(rhs),
            vec![self.clone()],
            Ops::Pow,
        );
        let self_grad = self.clone_grad();
        let self_data = self.clone_data();
        let out_grad = out.clone_grad();
        let back = Box::new(move || {
            self_grad.set(
                self_grad.get()
                    + (rhs * self_data.get().powf(rhs - 1.0)) * out_grad.get(),
            )
        }) as Box<dyn Fn() -> ()>;
        out.set_backward(back);
        out
    }

    pub fn relu(&self) -> Self {
        let out = if self.get_data() >= 0.0 {
            Self::_new(self.get_data(), vec![self.clone()], Ops::ReLU)
        } else {
            Self::_new(0.0, vec![self.clone()], Ops::ReLU)
        };
        let self_grad = self.clone_grad();
        let out_data = out.clone_data();
        let out_grad = out.clone_grad();
        let back = Box::new(move || {
            self_grad.set(
                self_grad.get()
                    + ((out_data.get() > 0.0) as u8 as f32) * out_grad.get(),
            )
        });
        out.set_backward(back);
        out
    }
}

impl Add<Self> for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Value::_new(
            self.get_data() + rhs.get_data(),
            vec![self.clone(), rhs.clone()],
            Ops::Add,
        );
        let self_grad = self.clone_grad();
        let rhs_grad = rhs.clone_grad();
        let out_grad = out.clone_grad();
        let back = Box::new(move || {
            self_grad.set(self_grad.get() + out_grad.get());
            rhs_grad.set(rhs_grad.get() + out_grad.get())
        }) as Box<dyn Fn() -> ()>;
        out.set_backward(back);
        out
    }
}

impl Add<Value> for &Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        self + &rhs
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, rhs: Value) -> Self::Output {
        &self + &rhs
    }
}

impl Add<&Value> for Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        &self + rhs
    }
}

impl Add<f32> for Value {
    type Output = Value;

    fn add(self, rhs: f32) -> Self::Output {
        let v = Value::new(rhs);
        &self + v
    }
}

impl Sub<Self> for &Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub<Self> for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        &self + &(-&rhs)
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Mul<Self> for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Value::_new(
            self.get_data() * rhs.get_data(),
            vec![self.clone(), rhs.clone()],
            Ops::Mul,
        );
        let self_grad = self.clone_grad();
        let self_data = self.clone_data();
        let rhs_grad = rhs.clone_grad();
        let rhs_data = rhs.clone_data();
        let out_grad = out.clone_grad();
        let back = Box::new(move || {
            self_grad.set(self_grad.get() + rhs_data.get() * out_grad.get());
            rhs_grad.set(rhs_grad.get() + self_data.get() * out_grad.get())
        }) as Box<dyn Fn() -> ()>;
        out.set_backward(back);
        out
    }
}

impl Mul<Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        self * &rhs
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Value) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<f32> for &Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        self * rhs
    }
}

impl Mul<f32> for Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = Value::new(rhs);
        &self * rhs
    }
}

impl Div<Self> for &Value {
    type Output = Value;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}

impl Div<f32> for &Value {
    type Output = Value;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<Value> for f32 {
    type Output = Value;

    fn div(self, rhs: Value) -> Self::Output {
        rhs.pow(-1.0) * self
    }
}

mod test {

    use super::*;
    #[test]
    fn test_add() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c + b;
        d.backward();
        assert_eq!(b.get_grad(), 2.0);
    }

    #[test]
    fn test_sub() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let d = c - b;
        d.backward();
        assert_eq!(b.get_grad(), -2.0);
    }

    #[test]
    fn test_mul() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c * b;
        d.backward();
        assert_eq!(b.get_grad(), 5.0);
    }

    #[test]
    fn test_mul_neg() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a - b;
        let d = c * b;
        d.backward();
        assert_eq!(b.get_grad(), -3.0);
    }

    #[test]
    fn test_pow() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c.pow(2.0);
        d.backward();
        assert_eq!(b.get_grad(), 6.0);
    }

    #[test]
    fn test_ReLU() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a + (b * 2.0);
        let d = c.relu();
        let e = d * 2.0;
        e.backward();
        assert_eq!(b.get_grad(), 4.0);
    }

    #[test]
    fn test_ReLU_neg() {
        let a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let c = a - (b * 2.0);
        let d = c.relu();
        let e = d * 2.0;
        e.backward();
        assert_eq!(b.get_grad(), 0.0);
    }

    #[test]
    fn test_div() {
        let ref a = Value::new(1.0);
        let ref b = Value::new(2.0);
        let ref c = a + b;
        let d = c / b;
        d.backward();
        assert_eq!(b.get_grad(), -0.25);
    }

    #[test]
    fn test_contrived() {
        let ref a = Value::new(-4.0);
        let ref b = Value::new(2.0);
        let mut c = a + b;
        let mut d = a * b + b.pow(3.0);
        c = c.clone() + c + 1.0;
        c = c.clone() + 1.0 + c.clone() + (-a);
        d = d.clone() + d.clone() * 2.0 + (b + a).relu();
        d = d.clone() + d.clone() * 3.0 + (b - a).relu();
        let e = c - d;
        let f = e.pow(2.0);
        let mut g = f.clone().div(2.0);
        g = g + 10.0 / f;
        assert_eq!(format!("{:.4}", g.get_data()), "24.7041");
        g.backward();
        assert_eq!(format!("{:.4}", a.get_grad()), "138.8338");
        assert_eq!(format!("{:.4}", b.get_grad()), "645.5773");
        // assert_eq!(b.get_grad(), -0.25);
    }
}
