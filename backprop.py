import math


# every Unit corresponds to a wire in the diagrams
class Unit:
    def __init__(self, value, grad):
        # value computed in the forward pass
        self.value = value
        # the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad = grad


class MultiplyGate:
    def forward(self, u0, u1):
        # store pointer to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)
        return self.utop

    def backward(self):
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad


class AddGate:
    def forward(self, u0, u1):
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop

    def backward(self):
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad


class SigmoidGate:
    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1 - s)) * self.utop.grad


if __name__ == '__main__':
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

    ax = mul_gate_0.forward(a, x)
    by = mul_gate_1.forward(b, y)

    axby = add_gate_0.forward(ax, by)
    axbyc = add_gate_1.forward(axby, c)
    s = sig_gate.forward(axbyc)

    s.grad = 1
    sig_gate.backward()
    add_gate_1.backward()
    add_gate_0.backward()
    mul_gate_1.backward()
    mul_gate_0.backward()
    print(a.grad, b.grad, c.grad, x.grad, y.grad)
