from drecexpbary import drecexpbary
import numpy as np

if __name__ == "__main__":
    oracle = lambda x: np.power(x, 2)
    x0 = np.array([0, 0])
    x = drecexpbary(oracle, x0, 10, 0.1, 0, 1, 100)
    
