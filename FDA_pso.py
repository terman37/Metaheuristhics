import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer

def fitnessFunction(x):
    sum = 0
    for i in range(0, x.shape[0] - 1):
        for j in range(i + 1, x.shape[0]):
            sum += (1 / (np.linalg.norm(x[i] - x[j])) ** 12) - (1 / (np.linalg.norm(x[i] - x[j])) ** 6)
    return sum

D = 10

x_max = 2 * np.ones(D)
x_min = -2 * np.ones(D)
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.8, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=65, dimensions=D, options=options, bounds=bounds)

# now run the optimization
cost, pos = optimizer.optimize(fitnessFunction, iters=100)

# cost evolution vs iterations
plot_cost_history(cost_history=optimizer.cost_history)

# m = Mesher(func=fitnessFunction, limits=[(-2, 2), (-2, 2)])
# d = Designer(limits=[(-2, 2), (-2, 2)], label=['x-axis', 'y-axis'])
# animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d)
# animation.save('plot0.gif', writer='imagemagick', fps=20)

plt.show()
