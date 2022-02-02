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

import sympy as sp  # pip install sympy
import random as rand
import numpy as np

# constants from assignment
LOOPS = 500
TRIALS = 10
STEPS = (0.1, 0.01, 0.001)
RAND = (-10, 10)

# set up the equation here
F = "x**2 - 6*x + y**2 + 10*y + 20"
SYMBOLS = ('x', 'y')
MIN = (3, -5)


# "silent"  does not print any output to the screen
# "quiet"   only prints results of final iteration of each trial
# "loud"    prints values after every single iteration
def gradient_descent(step_size, display="loud"):

    # initialize symbols
    symbols = []
    for s in SYMBOLS:
        symbols.append(sp.Symbol(s))

    # differentiate F with respect to x and y
    # del = gradient = vector derivative
    del_f = [sp.diff(F, i) for i in symbols]

    # initialize v to random values in range
    v = []
    for _ in range(len(symbols)):
        v.append(rand.uniform(RAND[0], RAND[1]))

    # helper function for displaying values
    def print_val(value, i, text):
        print(text + " " + str(i) + ":\t" + str(value[0]) + ", " + str(value[1]))

    # step function to implement the equation x_t = x_(t-1) - η∇f( x_(t-1) )
    def step(value):
        for i in range(len(symbols)):
            value[i] = value[i] - step_size * sp.N(del_f[i].subs(symbols[i], value[i]))
        return value

    # run GD for 500 steps
    for i in range(LOOPS):
        if display == "loud":
            print_val(v, i, "Iteration")
        v = step(v)

    # print and return final value
    if display != "silent":
        print_val(v, LOOPS, "Final Iteration")
    return v


# run the Gradient Descent algorithm and display results
def run():
    results = np.empty((len(STEPS), TRIALS, len(SYMBOLS)))

    for trial in range(TRIALS):
        print("Trial " + str(trial + 1) + "... ")
        for step in range(len(STEPS)):
            values = gradient_descent(STEPS[step], "silent")
            results[step][trial] = values

    # helper function to calculate distance from true minimum (3, -5)
    def distance(val):
        squared = 0
        for i in range(len(val)):
            squared += (val[i] - MIN[i])**2
        return squared**0.5

    # print out the results
    print("\nResults:\n")
    for step in range(len(STEPS)):
        total = avg = best = np.zeros(len(SYMBOLS))

        # find the best trial result for each η
        for trial in range(TRIALS):
            value = results[step][trial]
            total += value
            if distance(value) < distance(best):
                best = value

        # use total to calculate average (not a requirement)
        for t in range(len(total)):
            avg[t] = total[t] / TRIALS

        print("Step size: " + str(STEPS[step]))
        print("\t Average value: " + str(avg))
        print("\t Best value: " + str(best))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
