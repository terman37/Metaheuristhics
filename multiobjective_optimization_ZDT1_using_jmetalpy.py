from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot

if __name__ == '__main__':
    problem = ZDT1()
    max_evaluations = 10000

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))

    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    plot_front.plot(front, label=algorithm.get_name()+'-'+problem.get_name())
