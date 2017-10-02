#%%

'''
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''

#%%

import numpy as np
from sklearn import linear_model
import math
from numpy.linalg import *
from numpy import transpose as t
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats as stats


#%%

studentname1 = 'Jan Hynek'
studentname2 = 'Štěpán Svoboda'
studentnumber1 = 1189
studentnumber2 = 1188  # ŠTĚPÁNE NEZAPOMEŃ ZMĚNIT ČÍSLO
np.random.seed(studentnumber1 + 10 * studentnumber2)
print('This is the Python output for Assignment 1, Advanced Econometrics 1 for students: \n' +
      '   (1) ' + studentname1 + ' - ' + str(studentnumber1) + '\n' +
      '   (2) ' + studentname2 + ' - ' + str(studentnumber2) + '\n\n' +
      '   Group n\n')
# global variables
# N_obs = 50
# r = 3
beta = 1


# sigma_z = 2
# gamma = 0.50
# ro = 0.5
# pi_1 = 0.1
# pi = t(np.matrix([pi_1] + [0] * (r - 1)))


def cov_matrix(sigma_z, gamma, ro, r):
    cov_matrix = np.identity(r + 2) * sigma_z
    cov_matrix[:, 0] = (r + 2) * [gamma]
    cov_matrix[0, :] = (r + 2) * [gamma]
    cov_matrix[0, 0] = 1
    cov_matrix[1, 1] = 1
    cov_matrix[0, 1] = ro
    cov_matrix[1, 0] = ro
    return np.matrix(cov_matrix)


# defined as three values = [u, v, epsilon]


def errors(cov_matrix, N_obs, r):
    errors_vector = np.random.multivariate_normal(mean=[0] * (r + 2),
                                                  cov=cov_matrix,
                                                  size=N_obs)
    return np.matrix(errors_vector)


def iteration(gamma, ro, pi_1, N_obs, sigma_z, r):
    pi = t(np.matrix([pi_1] + [0] * (r - 1)))
    error_matrix = errors(cov_matrix(sigma_z, gamma, ro, r), N_obs, r)
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
    # we do not fit intercept as we know that our original model does not have intercept.
    # however, we think that in monte carlo simulation it wont make a difference if the intercept would be zero
    # as it is estimated as almost zero
    regression_ols = linear_model.LinearRegression(fit_intercept=False)
    regression_ols.fit(X=X, y=y)
    b_hat_ols = regression_ols.coef_
    # b_hat_ols2 = inv(t(X) * X) * t(X) * y
    # print(b_hat_ols - b_hat_ols2)
    u_hat_ols = y - X * b_hat_ols
    Omega_hat_ols = np.diag(np.power(u_hat_ols.A1, 2))
    Var_b_ols = (inv(t(X) * X) * t(X) *
                 Omega_hat_ols *
                 X * inv(t(X) * X))

    ###########################
    # Two stage least squares #
    ###########################

    # first regression
    regression_2sls_1 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_1.fit(X=Z, y=X)
    Pi_hat = t(np.matrix([regression_2sls_1.coef_[0][0], 0, 0]))
    X_hat = Z * Pi_hat

    # second regression
    regression_2sls_2 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_2.fit(X=X_hat, y=y)
    b_hat_2sls = regression_2sls_2.coef_
    u_hat_2sls = y - X * b_hat_2sls

    # obtaining variance
    Omega_hat_2sls = np.diag(np.power(u_hat_2sls.A1, 2))

    # S_hat = (t(Z) * Omega_hat_2sls * Z) / N_obs
    Var_b_2sls = (inv(t(X_hat) * X_hat) * t(X_hat) *
                  Omega_hat_2sls *
                  X_hat * inv(t(X_hat) * X_hat))

    # formatting results:
    result = (b_hat_ols[0, 0], b_hat_2sls[0, 0],
              Var_b_ols[0, 0], Var_b_2sls[0, 0], (pi.T * Z.T * Z * pi) 
              / (np.sum(np.power(error_matrix[:, 1], 2)) / N_obs))
    return [np.around(i, 3) for i in result]
    # print('b_hat_ols:  ' + str(b_hat_ols))
    # print('b_hat_2sls: ' + str(b_hat_2sls))
    # print('Var_b_ols:  ' + str(Var_b_ols))
    # print('Var_b_2sls: ' + str(Var_b_2sls))



#%%

def monte_carlo(gamma, ro, pi_1, n_iterations, N_obs, sigma_z, r):
    results_Hausman = np.zeros((n_iterations, 2))
    for i in range(n_iterations):
        (b_hat_ols, b_hat_2sls, Var_b_ols, Var_b_2sls, conc_par) = iteration(gamma, ro, pi_1, N_obs, sigma_z, r)
        d_Var = Var_b_2sls - Var_b_ols
        if d_Var == 0:
            d_Var_psinv = 0
        else:
            d_Var_psinv = 1 / d_Var

        Hausman = ((b_hat_ols - b_hat_2sls)) ** 2 * d_Var_psinv
        results_Hausman[i, 0] = Hausman
        df = b_hat_2sls[np.abs(b_hat_2sls) < 1e8].size
        pval = stats.chi2.sf(Hausman, df)
        results_Hausman[i, 1] = pval
    return (results_Hausman, conc_par)


#%%

Hausman_res1 = monte_carlo(gamma=0,
                          ro=0.5,
                          pi_1=0.4,
                          n_iterations=1000,
                          N_obs=100,
                          sigma_z=2,
                          r=3)


positive_res = len(np.where(Hausman_res1[0][:, 0] > stats.chi2.ppf(0.95, 1))[0])
print('The Hausman test passed the critical value', positive_res, 'times')
print('The concentration parameter is', Hausman_res1[1])

plt.hist(Hausman_res1[0][:, 0], bins=40)
plt.show()


#%%

Hausman_res2 = monte_carlo(gamma=0,
                          ro=0.5,
                          pi_1=15,
                          n_iterations=1000,
                          N_obs=100,
                          sigma_z=2,
                          r=3)

print(len(np.where(Hausman_res2[0][:, 0] > stats.chi2.ppf(0.95, 1))[0]))
print('The concentration parameter is', Hausman_res2[1])

plt.hist(Hausman_res2[0][:, 0], bins=40)
plt.show()

#%%
    pi_1 = 1
    r = 3
    sigma_z = 2
    gamma = 0
    N_obs = 100
    ro = 0.5
    pi = t(np.matrix([pi_1] + [0] * (r - 1)))
    error_matrix = errors(cov_matrix(sigma_z, gamma, ro, r), N_obs, r)
    # u = error_matrix[:, 0]
    # v = error_matrix[:, 1]

    Z = error_matrix[:, 2:]
    X = Z * pi + error_matrix[:, 1]
    y = X * beta + error_matrix[:, 0]

   

    ###########################
    # Two stage least squares #
    ###########################

    # first regression
    regression_2sls_1 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_1.fit(X=Z, y=X)
    Pi_hat = t(np.matrix([regression_2sls_1.coef_[0][0], 0, 0]))