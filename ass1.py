#%%



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

N = 20
r = 3
beta = 1
cov_matrix2 = cov_matrix2 * sigma_z
cov_matrix2[:, 0] = (r + 2) * [gamma]
cov_matrix2[0, :] = (r + 2) * [gamma]
cov_matrix2[0, 0] = 1
cov_matrix2[1, 1] = 1
cov_matrix2[0, 1] = ro
cov_matrix2[1, 0] = ro
print(cov_matrix2)



# defined as three values = [u, v, epsilon]

errors_vector = np.random.multivariate_normal(mean=[0, 0, 0],
                                              cov=cov_matrix,
                                              size=N)

z = errors_vector[:,2]

print(z)
