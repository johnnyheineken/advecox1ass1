import numpy as np
from sklearn import linear_model
import math
from numpy.linalg import *
from numpy import transpose as t
from itertools import chain
import matplotlib.pyplot as plt


studentname1 = 'Jan Hynek'
studentname2 = 'Štěpán Svoboda'
studentnumber1 = 1189
studentnumber2 = 1188  # ŠTĚPÁNE NEZAPOMEŃ ZMĚNIT ČÍSLO
np.random.seed(studentnumber1 + 10 * 1488)
print('This is the Python output for Assignment 1, Advanced Econometrics 1 for students: \n' +
      '   (1) ' + studentname1 + ' - ' + str(studentnumber1) + '\n' +
      '   (2) ' + studentname2 + ' - ' + str(studentnumber2) + '\n\n' +
      '   Group n\n')
# global variables
N = 50
r = 3
beta = 1


sigma_z = 2
gamma = 0.50
ro = 0.5
pi_1 = 0.1
pi = t(np.matrix([pi_1] + [0] * (r - 1)))


def cov_matrix(sigma_z, gamma, ro):
    cov_matrix = np.identity(r + 2) * sigma_z
    cov_matrix[:, 0] = (r + 2) * [gamma]
    cov_matrix[0, :] = (r + 2) * [gamma]
    cov_matrix[0, 0] = 1
    cov_matrix[1, 1] = 1
    cov_matrix[0, 1] = ro
    cov_matrix[1, 0] = ro
    return np.matrix(cov_matrix)


# defined as three values = [u, v, epsilon]


def errors(cov_matrix):
    errors_vector = np.random.multivariate_normal(mean=[0] * (r + 2),
                                                  cov=cov_matrix,
                                                  size=N)
    return np.matrix(errors_vector)


def monte_carlo(sigma_z, gamma, ro, pi):
    error_matrix = errors(cov_matrix(sigma_z, gamma, ro))
    # u = error_matrix[:, 0]
    # v = error_matrix[:, 1]

    Z = error_matrix[:, 2:]
    X = Z * pi + error_matrix[:, 1]
    y = X * beta + error_matrix[:, 0]

    #####################
    #####################
    ## estimating part ##
    #####################
    #####################

    ##########################
    # Ordinary least squares #
    ##########################
    regression_ols = linear_model.LinearRegression(fit_intercept=False)
    regression_ols.fit(X=X, y=y)
    b_hat_ols = regression_ols.coef_
    u_hat_ols = y - X * b_hat_ols
    Omega_hat_ols = np.diag(u_hat_ols.A1)
    Var_b_ols = (inv(t(X) * X) * t(X) *
                 Omega_hat_ols *
                 X * inv(t(X) * X))

    ###########################
    # Two stage least squares #
    ###########################

    # first regression
    regression_2sls_1 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_1.fit(X=Z, y=X)
    Pi_hat = t(np.matrix(regression_2sls_1.coef_))
    X_hat = Z * Pi_hat

    # second regression
    regression_2sls_2 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_2.fit(X=X_hat, y=y)
    b_hat_2sls = regression_2sls_2.coef_
    u_hat_2sls = y - X * b_hat_2sls

    # obtaining variance
    Omega_hat_2sls = np.diag(u_hat_2sls.A1)

    # S_hat = (t(Z) * Omega_hat_2sls * Z) / N
    Var_b_2sls = (inv(t(X_hat) * X_hat) * t(X_hat) *
                  Omega_hat_2sls *
                  X_hat * inv(t(X_hat) * X_hat))
    result = [b_hat_ols[0, 0], b_hat_2sls[0, 0],
              Var_b_ols[0, 0], Var_b_2sls[0, 0]]
    return [round(i, 2) for i in result]
    # print('b_hat_ols:  ' + str(b_hat_ols))
    # print('b_hat_2sls: ' + str(b_hat_2sls))
    # print('Var_b_ols:  ' + str(Var_b_ols))
    # print('Var_b_2sls: ' + str(Var_b_2sls))


LOOPS = 20
results = np.zeros((LOOPS, 4))
for i in range(LOOPS):
    results[i,:] = monte_carlo(sigma_z, gamma, ro, pi)

print(results)

