import os, sys
import importlib

from numpy.linalg import norm
from numpy.random import normal
from numpy import array

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir+'/baryopt')

import drecexpbary
import dbatchbary

importlib.reload(drecexpbary)
importlib.reload(dbatchbary)

oracle = lambda x: norm(x)

# Hyperparameters
nu = 5
sigma = 0.5
zeta = 0
lambda_ = 1
n_iters = 1000

x0 = array([1, 1])

# Recursive run
xhat_batch = drecexpbary.drecexpbary(oracle, x0, nu, sigma, zeta, lambda_, n_iters)

# Batch points for batch barycenter version
mu_x = 0
sigma_x = 1
size_x = [1000, 2]

xs = normal(mu_x, sigma_x, size_x)

xhat_recursive = dbatchbary.dbatchbary(oracle, xs, 10)

print("Batch     : "+str(xhat_batch))
print("Recursive : "+str(xhat_recursive))

