import numpy as np
from spirit.filtration import _ripser
from scipy.spatial.distance import pdist, squareform
from math import comb

np.random.seed(1234)
X = np.random.uniform(size=(50, 2))

# _ripser.do_something()

## Returns p-simplices & (p+1)-simplices to reduce
D = pdist(X) 
simplices, cols_to_reduce = _ripser.construct(D, 1, 1.5) # 105 cols to reduce

## (p = 1) <-> cols_to_reduce == edges that do not have zero apparent cofacet
## => they could be linearly independent; the other ones either: 
## a) have zero apparent cofacets and thus zero in R1, or 
## b) exist in dgm0 as destroyers

## (p = 2) <-> 


from comb_laplacian import flag_simplices
DM = squareform(D)
# er = 2 * np.min(DM.max(axis=1))
triangles = flag_simplices(D, 2, eps=1.5, verbose=True, discard_ap=True, n_blocks=8, shortcut=False)



