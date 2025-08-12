import numpy as np
import time


def BOA(population, objective_function, lb, ub, max_iterations):
    population_size, dimensions = population.shape
    fitness = np.array([objective_function(individual) for individual in population])

    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    Convergence = np.zeros(max_iterations)
    ct = time.time()

    # Main loop: Iterate through the number of iterations
    for t in range(max_iterations):
        mean_solution = np.mean(population, axis=0)
        worst_index = np.argmax(fitness)
        worst_solution = population[worst_index]
        # Iterate over the population
        for i in range(population_size):
            # Calculate the amount of Botox injection
            if t < max_iterations / 2:
                botox_amount = mean_solution - population[i]
            else:
                botox_amount = best_solution - population[i]

                # Scale and clip the botox_amount
            botox_amount = np.clip(botox_amount, lb[i], ub[i])
            new_position = population[i] + botox_amount
            new_position = np.clip(new_position, lb[i], ub[i])
            new_fitness = objective_function(new_position)
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = new_position
        Convergence[t] = best_fitness
    ct = time.time() - ct
    return best_fitness, Convergence, best_solution, ct
