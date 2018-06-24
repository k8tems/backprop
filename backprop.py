# every Unit corresponds to a wire in the diagrams
class Unit:
    def __init__(self, value, grad):
        # value computed in the forward pass
        self.value = value
        # the derivative of circuit output w.r.t this unit, computed in backward pass
        self.grad = grad
