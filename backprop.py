# Increase the output of a single neuron
# Taken from http://karpathy.github.io/neuralnets/ and refactored a bit
import math


class Unit:
    def __init__(self, value, grad):
        # value computed in the forward pass
        self.value = value
        # the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad = grad


class MultiplyGate:
    def forward(self, in0, in1):
        self.in0 = in0
        self.in1 = in1
        self.out = Unit(in0.value * in1.value, 0.0)
        return self.out

    def backward(self):
        # `out.grad` is the gradient of the outer function
        # cf. Chain rule: outer gradient w.r.t inner function * inner gradient
        # In this case, self.in*.grad is the inner and out.grad is the outer gradient
        # ∂/∂in0[in0*in1]=in1
        # ∂/∂in1[in0*in1]=in0
        self.in0.grad += self.in1.value * self.out.grad
        self.in1.grad += self.in0.value * self.out.grad


class AddGate:
    def forward(self, in0, in1):
        self.in0 = in0
        self.in1 = in1
        self.out = Unit(self.in0.value + self.in1.value, 0.0)
        return self.out

    def backward(self):
        # ∂/∂in0[in0+in1]=1
        # ∂/∂in1[in0+in1]=1
        self.in0.grad += 1 * self.out.grad
        self.in1.grad += 1 * self.out.grad


class SigmoidGate:
    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, in0):
        self.in0 = in0
        self.out = Unit(self.sig(self.in0.value), 0.0)
        return self.out

    def gradient(self, x):
        # The gradient of σ is weird because the output is used in its calculation
        # i.e. ∂σ/∂x=σ(x)(1-σ(x))
        return self.sig(x) * (1 - self.sig(x))

    def backward(self):
        self.in0.grad += self.gradient(self.in0.value) * self.out.grad


if __name__ == '__main__':
    """
    Calculate the gradient of σ(ax+by+c) and increase the result
    
    ∂σ/∂a = ∂σ/∂(ax+by+c) * ∂(ax+by+c)/∂a
    ∂(ax+by+c)/∂a = ∂(ax+by+c)/∂(ax+by) * ∂(ax+by)/∂a
    ∂(ax+by)/∂a = ∂(ax+by)/∂(ax) * ∂(ax)/∂a 
    
    Notice the chain rule is being applied backwards,
    with the gradient of the outer most function calculated first. 
    """
    a = Unit(1.0, 0.0)
    b = Unit(2.0, 0.0)
    c = Unit(-3.0, 0.0)
    x = Unit(-1.0, 0.0)
    y = Unit(3.0, 0.0)

    mul_gate_0 = MultiplyGate()
    mul_gate_1 = MultiplyGate()
    add_gate_0 = AddGate()
    add_gate_1 = AddGate()
    sig_gate = SigmoidGate()

    def forward_prop():
        ax = mul_gate_0.forward(a, x)
        by = mul_gate_1.forward(b, y)
        axby = add_gate_0.forward(ax, by)
        axbyc = add_gate_1.forward(axby, c)
        return sig_gate.forward(axbyc)
    s = forward_prop()
    print('initial result', s.value)

    def backward_prop():
        s.grad = 1
        sig_gate.backward()
        add_gate_1.backward()
        add_gate_0.backward()
        mul_gate_1.backward()
        mul_gate_0.backward()
        print(a.grad, b.grad, c.grad, x.grad, y.grad)

        step_size = 0.01
        a.value += step_size * a.grad
        b.value += step_size * b.grad
        c.value += step_size * c.grad
        x.value += step_size * x.grad
        y.value += step_size * y.grad
    backward_prop()

    # should be higher than initial value
    print('updated value', forward_prop().value)
