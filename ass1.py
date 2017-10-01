#%%

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)

#%%

import numpy as np
from sklearn import linear_model
import math
from numpy.linalg import *
from numpy import transpose as t
import matplotlib.pyplot as plt


#%%

studentname1 = 'Jan Hynek'
studentname2 = 'Štěpán Svoboda'
studentnumber1 = 1189
studentnumber2 = 1188  # ŠTĚPÁNE NEZAPOMEŃ ZMĚNIT ČÍSLO
np.random.seed(studentnumber1 + 10 * 1488)
print('This is the Python output for Assignment 1, Advanced Econometrics 1 for students: \n' +
      '   (1) ' + studentname1 + ' - ' + str(studentnumber1) + '\n' +
      '   (2) ' + studentname2 + ' - ' + str(studentnumber2) + '\n\n' +
      '   Group n')

N = 50
r = 3
beta = 1


sigma_z = 2
gamma = 0.50
ro = 0.5


def cov_matrix(sigma_z, gamma, ro):
    cov_matrix = np.identity(r + 2) * sigma_z
    cov_matrix[:, 0] = (r + 2) * [gamma]
    cov_matrix[0, :] = (r + 2) * [gamma]
    cov_matrix[0, 0] = 1
    cov_matrix[1, 1] = 1
    cov_matrix[0, 1] = ro
    cov_matrix[1, 0] = ro
    return cov_matrix


# defined as three values = [u, v, epsilon]

def errors(cov_matrix):
    errors_vector = np.random.multivariate_normal(mean=[0] * (r + 2),
                                                  cov=cov_matrix,
                                                  size=N)
    return errors_vector


z = errors(cov_matrix(sigma_z, gamma, ro))[:, 2:]



