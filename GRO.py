import numpy as np
import time


def bound_constraint(X_new, positions_i, lb, ub):
    return np.maximum(np.minimum(X_new, ub), lb)


# Gold Rush Optimizer (GRO)
def GRO(Positions, fobj, lb, ub, Max_iter):
    [N, dim] = Positions.shape
    lb = np.array(lb * dim)
    ub = np.array(ub * dim)

    sigma_initial = 2
    sigma_final = 1 / Max_iter

    best_pos = np.zeros(dim)
    best_score = np.inf  # change this to -inf for maximization problems

    Fit = np.full(N, np.inf)

    X_NEW = Positions.copy()
    Fit_NEW = Fit.copy()

    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = np.min(Fit)
    ct = time.time()
    iter = 1

    while iter <= Max_iter:
        for i in range(N):
            Fit_NEW[i] = fobj(X_NEW[i, :])

            if Fit_NEW[i] < Fit[i]:
                Fit[i] = Fit_NEW[i]
                Positions[i, :] = X_NEW[i, :]

            if Fit[i] < best_score:
                best_score = Fit[i]
                best_pos = Positions[i, :]

        l2 = ((Max_iter - iter) / (Max_iter - 1)) ** 2 * (sigma_initial - sigma_final) + sigma_final
        l1 = ((Max_iter - iter) / (Max_iter - 1)) ** 1 * (sigma_initial - sigma_final) + sigma_final

        for i in range(N):
            coworkers = np.random.choice(np.delete(np.arange(N), i), 2, replace=False)
            digger1, digger2 = coworkers

            m = np.random.rand()

            if m < 1 / 3:
                for d in range(dim):
                    r1 = np.random.rand()
                    D3 = Positions[digger2, d] - Positions[digger1, d]
                    X_NEW[i, d] = Positions[i, d] + r1 * D3
            elif m < 2 / 3:
                for d in range(dim):
                    r1 = np.random.rand()
                    A2 = 2 * l2 * r1 - l2
                    D2 = Positions[i, d] - Positions[digger1, d]
                    X_NEW[i, d] = Positions[digger1, d] + A2 * D2
            else:
                for d in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    C1 = 2 * r2
                    A1 = 1 + l1 * (r1 - 1 / 2)
                    D1 = C1 * best_pos[d] - Positions[i, d]
                    X_NEW[i, d] = Positions[i, d] + A1 * D1

            X_NEW = bound_constraint(X_NEW[i, :], Positions[i, :], lb, ub)

        Convergence_curve[iter-1] = best_score
        iter += 1
    ct = time.time() - ct
    return best_score, Convergence_curve, best_pos, ct
