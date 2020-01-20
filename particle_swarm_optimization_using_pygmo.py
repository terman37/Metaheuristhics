
from pygmo import *
import numpy as np
import matplotlib.pyplot as plt

class py_rosenbrock:
    def __init__(self,dim):
        self.dim = dim
    def fitness(self,x):
        global costs
        retval = np.zeros((1,))
        for i in range(len(x) - 1):
            retval[0] += 100.*(x[i + 1]-x[i]**2)**2+(1.-x[i])**2
        return retval
    def get_bounds(self):
        return (np.full((self.dim,),-5.),np.full((self.dim,),10.))

algo = algorithm(pso(gen = 100, omega=1,eta1=1.9,eta2=2.05,variant=3,max_vel=0.5 ))
algo.set_verbosity(1)
prob = problem(py_rosenbrock(4))
pop = population(prob, 10)

pop = algo.evolve(pop)

# plt.plot(costs)
# plt.show()

best_fitness = pop.get_f()[pop.best_idx()]
costs = pop.get_f()
pos = pop.get_x()
it = pop.get_ID()

plt.plot(it)
plt.show()

print(it,best_fitness)
print(pop.champion_x)