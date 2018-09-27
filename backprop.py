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
        # ∂/∂in0[in0*in1]*out.grad=in1*out.grad
        # ∂/∂in1[in0*in1]*out.grad=in0*out.grad
        self.in0.grad += self.in1.value * self.out.grad
        self.in1.grad += self.in0.value * self.out.grad


class AddGate:
    def forward(self, in0, in1):
        self.in0 = in0
        self.in1 = in1
        self.out = Unit(self.in0.value + self.in1.value, 0.0)
        return self.out

    def backward(self):
        # ∂/∂in0[in0+in1]*out.grad=1*out.grad
        # ∂/∂in1[in0+in1]*out.grad=1*out.grad
        self.in0.grad += 1 * self.out.grad
        self.in1.grad += 1 * self.out.grad


class SigmoidGate:
    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, in0):
        self.in0 = in0
        self.out = Unit(self.sig(self.in0.value), 0.0)
        return self.out

    def backward(self):
        s = self.sig(self.in0.value)
        # This is the final gate so the gradient for the output is 1
        # (i.e. d/dx[x]=1)
        # ∂σ/∂x*d/dx[x]=σ(1-σ)*1
        self.in0.grad += (s * (1 - s)) * self.out.grad


if __name__ == '__main__':
    """Calculate the gradient of ax+by+c and increase the result"""
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
