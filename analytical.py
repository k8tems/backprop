# Compute gradient with the analytical method and
# compare with the gradient produced via the numerical method as a sanity check
import math


def forwardCircuitFast(a,b,c,x,y):
  return 1/(1 + math.exp( - (a*x + b*y + c)))


a = 1
b = 2
c = -3
x = -1
y = 3
h = 0.0001
a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h
x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h
y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h
print(a_grad, b_grad, c_grad, x_grad, y_grad)
