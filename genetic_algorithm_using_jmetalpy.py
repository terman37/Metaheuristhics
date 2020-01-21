import numpy as np
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BestSolutionSelection, SimpleRandomMutation, SBXCrossover
from jmetal.core.problem import FloatProblem, FloatSolution
from jmetal.util.termination_criterion import StoppingByEvaluations

class MyRosen(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(MyRosen, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5. for _ in range(number_of_variables)]
        self.upper_bound = [10. for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        result = 0
        x = solution.variables
        for i in range(solution.number_of_variables - 1):
            result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        solution.objectives[0] = result
        return solution

    def get_name(self) -> str:
        return 'MyRosen'

problem = MyRosen(5)

algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=50,
    mutation=SimpleRandomMutation(0.4),
    crossover=SBXCrossover(0.9, 20.0),
    selection=BestSolutionSelection(),
    termination_criterion=StoppingByEvaluations(max=5000)
)

algorithm.run()
result = algorithm.get_result()

print('Algorithm: {}'.format(algorithm.get_name()))
print('Problem: {}'.format(problem.get_name()))
print('Solution: {}'.format(result.variables))
print('Fitness: {}'.format(result.objectives[0]))
print('Computing time: {}'.format(algorithm.total_computing_time))

