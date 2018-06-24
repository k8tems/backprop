# every Unit corresponds to a wire in the diagrams
class Unit:
    def __init__(self, value, grad):
        # value computed in the forward pass
        self.value = value
        # the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad = grad


class multiplyGate:
  def forward(self, u0, u1):
      # store pointer to input Units u0 and u1 and output unit utop
      self.u0 = u0
      self.u1 = u1
      self.utop = Unit(u0.value * u1.value, 0.0)
      return self.utop

  def backward(self):
      self.u0.grad += self.u1.value * self.utop.grad
      self.u1.grad += self.u0.value * self.utop.grad
