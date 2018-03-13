###################################################################
###################For outlier OD estimation using RPCA############
###################################################################
import numpy as np
from numpy import linalg as LA
import math




def rpca(M, tol = 1e-5, delta = 1e-5, eita = 0.9, lambda_rpca = None, mu = None, maxiter = 1000):
    if mu is None:
        mu = 0.99*LA.norm(M, 2)
    m, n = M.shape
    if lambda_rpca is None:
        lambda_rpca = 0.05
    convergence = tol*LA.norm(M, 2)
    #Initializing the matrices
    L_1 = np.zeros(M.shape)
    L = np.zeros(M.shape)
    S_1 = np.zeros(M.shape)
    S = np.zeros(M.shape)

    t_1 = 1
    t = 1
    mu_bar = delta*mu
    iter = 0
    while iter < maxiter:
        Y_L = L + ((t_1 - 1)/t)*(L - L_1)
        Y_S = S + ((t_1 - 1)/ t) * (S - S_1)
        G_L = Y_L  - 0.5*(Y_L + Y_S - M)
        G_S = Y_S  - 0.5*(Y_L + Y_S - M)
        U, sigma, V = LA.svd(G_L)
        L_1 = L
        S_1 = S
        f,g = U.shape
        j,k = V.shape
        sig = np.zeros((g, j))
        sig[:len(sigma), :len(sigma)] = np.diag(sigma)
        L =  np.dot(U, np.dot(svt(sig, mu/2), V))
        S = svt(G_S, mu*lambda_rpca*0.5)
        t_1 = t
        t = (1+ math.sqrt(4*(t**2) + 1))*0.5
        mu = max(eita*mu, mu_bar)
        iter = iter + 1
        print(LA.matrix_rank(L_1))
    return L_1, S_1






def svt(X, tau):
    X[abs(X) < tau] = 0.0
    sgn = np.sign(X)
    X = X - sgn*tau
    return X











import numpy as np
import csv
import pandas as pd

#Please provide flow matrix M mentioned in the paper
M = pd.read_csv('od.csv', sep = ',', header=None, encoding='utf-8-sig')

L, S = rpca(M.as_matrix(), lambda_rpca = 0.09)



np.savetxt('L_1.csv', L, delimiter=',')
np.savetxt('S_1.csv', S, delimiter=',')




