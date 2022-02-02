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
import sympy as sp
import random as rand
import numpy as np

LOOPS = 500  # default 500
TRIALS = 10  # default 10
STEPS = (0.1, 0.01, 0.001)  # default 0.1, 0.01, 0.001


# "silent"  does not print any output to the screen
# "quiet"   only prints results of final iteration of each trial
# "loud"    prints values after every single iteration
def gradient_descent(step_size, display="loud"):

    # initialize symbols
    x, y = sp.symbols('x y')
    # x = sp.Symbol('x') # singular

    # create the equation
    # f(x, y) = x^2 - 6x + y^2 + 10y + 20
    f = x**2 - 6*x + y**2 + 10*y + 20

    # differentiate f with respect to x and y
    del_f = [sp.diff(f, i) for i in (x, y)]  # "del", gradient, vector derivative

    # initialize v to random values in range
    v = [rand.uniform(-10, 10), rand.uniform(-10, 10)]

    # set a step size
    eta = step_size

    def print_val(value, i, text):
        print(text + " " + str(i) + ":\t" + str(value[0]) + ", " + str(value[1]))

    # create function for incrementing X
    def step(value):
        # x_t = x_(t-1) - η∇f( x_(t-1) )
        for i in range(len(value)):
            # eta is just in scope
            # I am substituting the value for x and y both times, out of confusion mostly
            value[i] = value[i] - eta * sp.N(del_f[i].subs(x, value[i]).subs(y, value[i]))
        return value

    # run gd for 500 steps
    for i in range(LOOPS):
        if display == "loud":
            print_val(v, i, "Iteration")
        v = step(v)

    if display != "silent":
        print_val(v, LOOPS, "Final Iteration")
    return v


# basic little function for running the program
def run():
    results = np.empty((len(STEPS), TRIALS, 2))

    for trial in range(TRIALS):
        print("Trial " + str(trial + 1) + "... ")

        for step in range(len(STEPS)):
            values = gradient_descent(STEPS[step], "silent")
            results[step][trial] = values

    def distance(val):
        # calculate distance from true minimum (3, -5)
        a = abs(val[0] - 3)
        b = abs(val[1] + 5)
        return pow(a**2 + b**2, 0.5)

    print("\nResults:\n")
    for step in range(len(STEPS)):
        total = [0, 0]
        avg = [0, 0]
        best = [0, 0]
        for trial in range(TRIALS):
            value = results[step][trial]
            total += value
            if distance(value) < distance(best):
                best = value

        avg[0] = total[0] / TRIALS
        avg[1] = total[1] / TRIALS

        print("Step size: " + str(STEPS[step]))
        print("\t Average value: " + str(avg))
        print("\t Best value: " + str(best))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
