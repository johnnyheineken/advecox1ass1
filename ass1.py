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
import pandas as pd


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
    Pi_hat = t(np.matrix(regression_2sls_1.coef_))
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
              Var_b_ols[0, 0], Var_b_2sls[0, 0], (pi.T * Z.T * Z * pi))

    return [np.around(i, 3) for i in result]
    # print('b_hat_ols:  ' + str(b_hat_ols))
    # print('b_hat_2sls: ' + str(b_hat_2sls))
    # print('Var_b_ols:  ' + str(Var_b_ols))
    # print('Var_b_2sls: ' + str(Var_b_2sls))


#%%

def monte_carlo(gamma, ro, pi_1, n_iterations, N_obs, sigma_z, r):
    results_Hausman = np.zeros((n_iterations, 3))
    for i in range(n_iterations):
        (b_hat_ols, b_hat_2sls, Var_b_ols, Var_b_2sls, conc_par) = iteration(
            gamma, ro, pi_1, N_obs, sigma_z, r)
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
        results_Hausman[i, 2] = conc_par
    return results_Hausman, [gamma, ro, pi_1, n_iterations, N_obs, sigma_z, r]


#%%


def outcome(gamma, ro, pi_1, n_iterations, N_obs, sigma_z, r):
    Hausman_res, parameters = monte_carlo(gamma, ro, pi_1, n_iterations, N_obs, sigma_z, r)
    used_pars=pd.DataFrame.from_items([('parameters:', parameters)], orient='index',
                                        columns=['gamma', 'ro', 'pi_1',
                                                 'n_iterations', 'N_obs',
                                                 'sigma_z', 'r'])
    print(used_pars)
    positive_res=len(
        np.where(Hausman_res[:, 0] > stats.chi2.ppf(0.95, 1))[0])

    print('\nThe Hausman test passed the critical value ' + str(positive_res) +
          ' times out of ' + str(parameters[3]) + ' iterations')
    print('The average concentration parameter is ' +
          str(np.mean(Hausman_res[:, 2])))

    plt.hist(Hausman_res[:, 0], bins=40)
    plt.show()


outcome(gamma=0,
        ro=0.2,
        pi_1=0.5,
        n_iterations=500,
        N_obs=50,
        sigma_z=2,
        r=3)



#%%


# np.array([[r, N_obs, ro, gamma, pi_1],
#           [3, 50, 0, 0, 0.5], # realistic assumption is we study overidentified case, let's keep 3
#           [1, 50, 0, 0, 0.5],
#           [5, 50, 0, 0, 0.5],
#           [3, 50, 0, 0, 0.1], # weak instrument. big variance, very few rejections
#           [3, 100, 0, 0, 0.5], # more observations means higher concentration parameter
#           [3, 100, 0.25, 0, 0.1],
#           [3, 100, 0.25, 0, 0.25],
#           [3, 100, 0.25, 0, 0.5],
#           [3, 100, 0.5, 0, 0.1],
#           [3, 100, 0.5, 0, 0.25],
#           [3, 100, 0.5, 0, 0.5],
#           [3, 100, 0.25, 0.1, 0.1], # show the risks of gamma > 0
#           [3, 100, 0.25, 0.1, 0.25], # if i were to assume z is exogenous then i would come to the conclusion x is exogenous as well
#           [3, 100, 0.25, 0.1, 0.5]]) # i cannot say anything
    
    
    
