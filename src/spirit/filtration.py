# import numpy as np
# import numba as nb 
# from math import comb

# import _ripser


# @nb.jit(parallel=True)
# def _rips_simplices(X: np.ndarray, dim: int, radius: float): 
#   n: int = len(X) 
#   include = np.array(comb(n, dim+1), dtype=bool)
#   for i in nb.prange(N):
#     include[i] = 