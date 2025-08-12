import numpy as np
import time


# Define the probability density function (p_obj)
def p_obj(x):
    return 0.2 * (1 / (np.sqrt(2 * np.pi) * 3)) * np.exp(-((x - 25) ** 2) / (2 * 3 ** 2))


# Crayfish Optimization Algorithm (COA)
def CFO(X, fobj, lb, ub, T):
    N, dim = X.shape
    cuve_f = np.zeros(T)
    global_Cov = np.zeros(T)
    Best_fitness = np.inf
    best_position = np.zeros(dim)
    fitness_f = np.zeros(N)
    ct = time.time()
    # Initial fitness calculation
    for i in range(N):
        fitness_f[i] = fobj(X[i, :])
        if fitness_f[i] < Best_fitness:
            Best_fitness = fitness_f[i]
            best_position = X[i, :].copy()

    global_position = best_position.copy()
    global_fitness = Best_fitness
    cuve_f[0] = Best_fitness

    t = 0
    while t < T:
        C = 2 - (t / T)  # Eq.(7)
        temp = np.random.rand() * 15 + 20  # Eq.(3)
        xf = (best_position + global_position) / 2  # Eq.(5)
        Xfood = best_position.copy()

        Xnew = np.zeros_like(X)

        for i in range(N):
            if temp > 30:
                # Summer resort stage
                if np.random.rand() < 0.5:
                    Xnew[i, :] = X[i, :] + C * np.random.rand(dim) * (xf - X[i, :])  # Eq.(6)
                else:
                    # Competition stage
                    for j in range(dim):
                        z = np.random.randint(0, N)  # Eq.(9)
                        Xnew[i, j] = X[i, j] - X[z, j] + xf[j]  # Eq.(8)
            else:
                # Foraging stage
                P = 3 * np.random.rand() * fitness_f[i] / fobj(Xfood)  # Eq.(4)
                if P > 2:  # The food is too big
                    Xfood = np.exp(-1 / P) * Xfood  # Eq.(12)
                    for j in range(dim):
                        Xnew[i, j] = (X[i, j] + np.cos(2 * np.pi * np.random.rand()) * Xfood[j] * p_obj(temp)
                                      - np.sin(2 * np.pi * np.random.rand()) * Xfood[j] * p_obj(temp))  # Eq.(13)
                else:
                    Xnew[i, :] = (X[i, :] - Xfood) * p_obj(temp) + p_obj(temp) * np.random.rand(dim) * X[i,
                                                                                                       :]  # Eq.(14)

        # Boundary conditions
        for i in range(N):
            Xnew[i, :] = np.clip(Xnew[i, :], lb[i], ub[i])

        global_position = Xnew[0, :].copy()
        global_fitness = fobj(global_position)

        for i in range(N):
            # Obtain the optimal solution for the updated population
            new_fitness = fobj(Xnew[i, :])
            if new_fitness < global_fitness:
                global_fitness = new_fitness
                global_position = Xnew[i, :].copy()

            # Update the population to a new location
            if new_fitness < fitness_f[i]:
                fitness_f[i] = new_fitness
                X[i, :] = Xnew[i, :].copy()
                if fitness_f[i] < Best_fitness:
                    Best_fitness = fitness_f[i]
                    best_position = X[i, :].copy()

        global_Cov[t] = global_fitness
        cuve_f[t] = Best_fitness
        t += 1

    CT = time.time() - ct
    return Best_fitness, cuve_f, best_position, CT



