import numpy as np
import math

class RobustPCA:
    def __init__(self, X, tol = 1e-5, max_iter=1000):
        self.X = X
        self.tol = tol
        self.max_iter = max_iter

    def S_lambda(self, lambda_, X):
        result = np.sign(X) * np.maximum(np.abs(X) - lambda_, np.zeros(X.shape))
        return result

    def D_lambda(self, lambda_, X):
        U, S, V_t = np.linalg.svd(X,full_matrices=0,compute_uv=1)
        S_d = np.diag(S)

        result = np.dot(np.dot(U, self.S_lambda(lambda_, S_d)), V_t)
        return result

    def iterate(self):
        M = self.X.shape[0]
        N = self.X.shape[1]
        lambda_ = 1 / math.sqrt(max(M, N)) / 3
        mu = 10 * lambda_

        normX = np.linalg.norm(self.X, "fro")
        print("normX: ", normX)

        L = np.zeros([M, N])
        S = np.zeros([M, N])
        Y = np.zeros([M, N])

        for i in range (1, self.max_iter):
            L = self.D_lambda(1/mu, self.X - S + (1/mu) * Y)

            S = self.S_lambda(lambda_/mu, self.X - L + (1/mu) * Y)
            Z = self.X - L - S

            Y = Y + mu * Z

            err = np.linalg.norm(Z, "fro") / normX
            
            print("iter: ", i, ", error: ", err, ", rank(L): ", np.linalg.matrix_rank(L))

            if (err < self.tol):
                break

        return L, S