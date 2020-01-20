import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from pyswarms.utils.plotters.formatters import Mesher, Designer

D = 2

x_max = 2 * np.ones(D)
x_min = -1 * np.ones(D)
bounds = (x_min, x_max)
options = {'c1': 0.3, 'c2': 0.1, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=D, options=options, bounds=bounds)

# now run the optimization
cost, pos = optimizer.optimize(fx.rosenbrock, 100)

# print(optimizer.pos_history)
m = Mesher(func=fx.rosenbrock, limits=[(-1, 2), (-1, 2)])
d = Designer(limits=[(-1, 2), (-1, 2)], label=['x-axis', 'y-axis'])
animation = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d)
animation.save('plot0.gif', writer='imagemagick', fps=20)

# pos_history_3d = m.compute_history_3d(optimizer.pos_history)  # preprocessing
# animation = plot_surface(pos_history=pos_history_3d,
#                             mesher=m, designer=d,
#                             mark=(1, 1, 1))
# animation.save('plot0.gif', writer='imagemagick', fps=10)

# cost evolution vs iterations
plot_cost_history(cost_history=optimizer.cost_history)

plt.show()
