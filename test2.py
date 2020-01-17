import math


def fitnessFunction(v):
    m = 1 + math.cos(0.04 * v) ** 2
    n = math.exp(-v ** 2 / (20000))
    return m * n


import numpy as np
import matplotlib.pyplot as plt

v = np.arange(-500., 500., .5)
fitness = []
for x in v:
    fitness.append(fitnessFunction(x))

plt.plot(v,fitness)
# plt.show()

import random


def simulated_annealing(init_state, t0, alpha=0.9, tend=0.01, max_nit=30):
    t = t0
    current_state = init_state
    oldbest = current_state
    s_st = []
    s_e = []
    while t > tend:
        nit = 1
        if fitnessFunction(current_state) > fitnessFunction(oldbest):
            oldbest = current_state
        else:
            current_state = oldbest
        while nit <= max_nit:
            e_current = fitnessFunction(current_state)
            s_e.append(e_current)
            s_st.append(current_state)
            next_state = current_state + perturbation()
            e_next = fitnessFunction(next_state)
            delta_e = e_next - e_current
            delta_e = -delta_e
            if delta_e < 0:
                current_state = next_state
            else:
                rv = random.random()
                if math.exp(-delta_e / t) > rv:
                    current_state = next_state
            nit += 1
        t = t * alpha
    plt.plot(s_st,s_e,"-o")
    return current_state

def perturbation():
    res = 50 * random.uniform(-1, 1)
    return res


init_value = 250
init_temp = 10

result = simulated_annealing(init_value, init_temp)
print(result)

plt.show()
