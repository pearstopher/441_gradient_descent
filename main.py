# CS441 homework 2 problem 13
# gradient descent

# Write a short program to execute the Gradient Descent (GD) algorithm as described in class.
# Recall that the key steps in GD are as follows:
#   x_0 = random
#   x_t = x_(t-1) - η∇f( x_(t-1) )
#
# Apply GD to approximately solve for the global minimum of the function
#   f(x, y) = x^2 - 6x + y^2 + 10y + 20
#
# You will run (3) sets of experiments using different values for η:
#   (i)     η = .1,
#   (ii)    η =.01, and
#   (iii)   η =.001.
#
# Run GD for 500 steps for each experiment; in each case initialize x_0 ∈ [−10,10] × [−10,10].
# Report the best performance out of 10 trials for each of the different η value cases.
# Provide some comments and analysis about your results.
# Please include your (concise) GD code in your assignment write-up.
#

# going to calculate a derivative!
# pip install sympy
# from sympy import *
import sympy as sp
import random as rand

LOOPS = 500


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # initialize symbols
    x, y = sp.symbols('x y')
    # x = sp.Symbol('x') # singular

    # create the equation
    # f(x, y) = x^2 - 6x + y^2 + 10y + 20
    f = x**2 - 6*x + y**2 + 10*y + 20

    # differentiate f with respect to x and y
    del_f = [sp.diff(f, i) for i in (x, y)]  # "del" or gradient

    # initialize X to random values in range
    X = [rand.uniform(-10, 10), rand.uniform(-10, 10)]

    # set a step size
    eta = 0.1

    def print_val(value, i, text):
        print(text + " " + str(i) + ":\t" + str(value[0]) + ", " + str(value[1]))

    # create function for incrementing X
    def step(value, step_size):
        # x_t = x_(t-1) - η∇f( x_(t-1) )
        for i in range(len(value)):
            value[i] = value[i] - step_size * sp.N(del_f[i].subs(x, value[i]).subs(y, value[i]))
        # return value - step_size * sp.N(del_f.subs(x, value[0]).subs(y, value[1]))
        return value


    # run gd for 500 steps
    for i in range(LOOPS):
        print_val(X, i, "Value")
        X = step(X, eta)

    print_val(X, LOOPS, "Final")

    # print(sp.N(f.subs(x, 1).subs(y, 1)))

    # c = sp.Symbol('c')
    # d = c * 5
    # d = d.subs(c, 6)
    # print(sp.N(d))

    # sp.Eq(a, a.doit())
    # print(b)