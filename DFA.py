import numpy as np
import time


# Dark Forest Algorithm (DFA)
def DFA(population, fitness_function, lower_bound, upper_bound, max_iter=50):
    population_size, dim = population.shape
    b, delta, beta = 0.5, 0.1, 0.9

    fitness_values = np.apply_along_axis(fitness_function, 1, population)

    # Initialize the best fitness and solution
    best_solution = population[np.argmin(fitness_values)]
    best_fitness = np.min(fitness_values)
    convergence = np.zeros(max_iter)
    CT = time.time()
    iter = 0
    while iter < max_iter:
        ranked_idx = np.argsort(fitness_values)
        population = population[ranked_idx]  # Sort population by fitness

        # Update the best solution and fitness
        if np.min(fitness_values) < best_fitness:
            best_solution = population[0]
            best_fitness = np.min(fitness_values)

        xf = (best_solution + population[0]) / 2  # Eq. (5)

        for i in range(population_size):
            temp = np.random.rand() * 15 + 20  # Eq. (3)
            C = 2 - (iter / max_iter)  # Eq. (7)

            if temp > 30:
                if np.random.rand() < 0.5:
                    # Summer resort stage - Eq. (6)
                    population[i] = population[i] + C * np.random.rand(dim) * (xf - population[i])
                else:
                    # Competition stage - Eq. (8)
                    for j in range(dim):
                        z = np.random.randint(0, population_size)
                        population[i, j] = population[i, j] - population[z, j] + xf[j]
            else:
                # Foraging stage - Eq. (4) and Eq. (12)-(14)
                P = 3 * np.random.rand() * fitness_values[i] / fitness_function(best_solution)
                if P > 2:
                    # Eq. (12)
                    best_solution = np.exp(-1 / P) * best_solution
                    for j in range(dim):
                        population[i, j] = population[i, j] + np.cos(2 * np.pi * np.random.rand()) * best_solution[j] \
                                           - np.sin(2 * np.pi * np.random.rand()) * best_solution[j]
                else:
                    # Eq. (14)
                    population[i] = (population[i] - best_solution) * np.random.rand() + best_solution

        # Apply boundary conditions
        population = np.clip(population, lower_bound, upper_bound)

        # Recalculate fitness
        fitness_values = np.apply_along_axis(fitness_function, 1, population)

        convergence[iter] = np.min(fitness_values)
        iter += 1
    # Refine best solution by adjusting with beta - Eq. (16)-(18)
    best_solution = best_solution - beta * np.random.rand(dim)
    best_fitness = fitness_function(best_solution)
    CT = time.time() - CT
    return best_fitness, convergence, best_solution, CT

