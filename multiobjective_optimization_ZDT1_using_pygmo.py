from pygmo import *
from matplotlib import pyplot as plt

D=10
udp = zdt(prob_id=6, param=D)
pop = population(prob=udp, size=100)

plt.figure(0)
ax = plot_non_dominated_fronts(pop.get_f())
plt.title("ZDT: random initial population")

uda = nsga2(gen=250, cr=0.99, eta_c=20, m=1/D, eta_m=20)
algo = algorithm(uda)

plt.figure(1)
pop = algo.evolve(pop)
ax2 = plot_non_dominated_fronts(pop.get_f())
plt.title("ZDT: ... and the evolved population")

plt.show()
