# Compute gradient with the analytical method and
# compare with the gradient produced via the numerical method as a sanity check
# Highly nonlinear equations are not possible to solve with the analytical method?
# https://www.researchgate.net/post/What_are_the_advantages_of_numerical_method_over_analyatical_method2
# -> Let f be a highly nonlinear equation
# -> x_1_grad = (f(x_1+h, x_2, ... x_n) + f(x_1, x_2, ... x_n)) / h
# Why won't this work??
import math


def circuit(a, b, c, x, y):
    return 1/(1 + math.exp(-(a*x + b*y + c)))


a = 1
b = 2
c = -3
x = -1
y = 3
h = 0.0001
a_grad = (circuit(a + h, b, c, x, y) - circuit(a, b, c, x, y)) / h
b_grad = (circuit(a, b + h, c, x, y) - circuit(a, b, c, x, y)) / h
c_grad = (circuit(a, b, c + h, x, y) - circuit(a, b, c, x, y)) / h
x_grad = (circuit(a, b, c, x + h, y) - circuit(a, b, c, x, y)) / h
y_grad = (circuit(a, b, c, x, y + h) - circuit(a, b, c, x, y)) / h
print(a_grad, b_grad, c_grad, x_grad, y_grad)
