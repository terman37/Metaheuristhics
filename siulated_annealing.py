import math
import random
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt

def SA(max_it, init_temp, alpha, final_temp, init_state):
    nit = 1
    # k = scipy.constants.Boltzmann
    current_state = init_state
    while nit <= max_it:
        t = init_temp
        nit += 1
        while t > final_temp:
            next_state = current_state + perturbation(nit)  # + perturbations
            energy_delta = myvalue(next_state) - myvalue(current_state)
            if energy_delta < 0 or math.exp(-energy_delta / t) > random.random():
                current_state = next_state
                addSolution(current_state)
                t = alpha * t
        print("%d %f" % (nit, t))
        print(current_state)
    print("finaltemp acheived in %d iterations" % nit)
    print(current_state)


def perturbation(v):
    val = v*50
    return np.random.uniform(-1 / val, 1 / val, 5)


def myvalue(x):
    # Rosen function
    sum = 0
    for i in range(x.shape[0] - 1):
        sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum


def myvalue_penalized(x):
    # function3 + penalty for constarints x >= -5 and x <= 10
    global ro
    ro = 1
    sum = myvalue(x)
    for i in range(x.shape[0]):
        if x[i] < -5:
            sum += ro * (-5 - x[i]) ** 2
            # sum += ro * (-1) / (-5 - x[i])
        elif x[i] > 10:
            sum += ro * (x[i] - 10) ** 2
            # sum += ro * (-1) / (x[i] - 10)
    return sum

def addSolution(xk):
    global testSolutions
    testSolutions.append(myvalue(xk))

global testolutions
testSolutions = []
x = np.random.uniform(-5, 5, 5)
print(x)
SA(1, 100000, 0.9999, 2.5, x)
plt.plot(testSolutions)
plt.show()
