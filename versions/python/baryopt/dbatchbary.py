def dbatchbary(oracle, xs, nu = 1, lambda_ = 1):
    ''' 
     Batch barycenter algorithm for direct optimization
     
     In:
       - oracle     [function]   : Oracle function e.g. lambda x: numpy.power(x, 2)
       - xs         [list[list]] : list with coordinates
       - nu         [double]     : positive value (Caution on its value due overflow)
       - lambda     [double]     : Forgetting factor between 0 and 1

     Out:
        - x [list]: Optimum position
    '''
    from functools import reduce
    from numpy import exp
    
    numerator = reduce(lambda LR, curr_x: lambda_*LR + curr_x*exp(-nu*oracle(curr_x)), xs)
    denominator = reduce(lambda LR, curr_x: lambda_*LR + exp(-nu*oracle(curr_x)), xs)
    
    return numerator/denominator
