from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


def function3(x):
    # Rosen function
    sum = 0
    for i in range(x.shape[0] - 1):
        sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return sum


def penalty_function(x):
    # function3 + penalty for constarints x >= -5 and x <= 10
    global ro
    sum = function3(x)
    for i in range(x.shape[0]):
        if x[i] < -5:
            sum += ro * (-5 - x[i]) ** 2
        elif x[i] > 10:
            sum += ro * (x[i] - 10) ** 2
    return sum


def addSolution(xk):
    global testSolutions
    testSolutions.append(function3(xk))


def addSolution2(xk):
    global testSolutions2
    testSolutions2.append(function3(xk))


def processProblem(D, myFunction):
    x = np.random.uniform(-20, 20, D)
    print(x)
    global testSolutions
    global testSolutions2
    global ro

    # Use Simplex without any penalties
    testSolutions = []
    res1 = optimize.minimize(function3, x, method='Nelder-Mead', tol=1e-4, callback=addSolution,
                             options={'maxiter': 2000})
    print("No penalty success = %s in %d iter" % (res1.success, res1.nit))
    print(res1.x)

    # Use Simplex with penaltized function
    testSolutions2 = []
    ro = 1
    totiter = 0
    stop = False
    i = 1
    while i < 50 and stop is False:
        print("i=%d ro=%f" % (i, ro))
        res2 = optimize.minimize(penalty_function, x, method='Nelder-Mead', tol=1e-4, callback=addSolution2,
                                 options={'maxiter': 2000})
        totiter += res2.nit
        if res2.success:
            stop = True
        x = res2.x
        ro += ro * 0.1
        i += 1
    print("With penalty success = %s in %d iter" % (res2.success, res2.nit))
    print("Total iterations: %d" % totiter)
    print(res2.x)

    # plt.plot(testSolutions)
    plt.plot(testSolutions2)
    plt.show()


# Optimize RosenBrock function with n dimensions
processProblem(20, function3)
