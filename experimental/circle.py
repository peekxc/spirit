
# %% 
import numpy as np 
import splex as sx
from itertools import combinations
from scipy.spatial.distance import pdist 
from spirit.apparent_pairs import clique_mod
from combin import comb_to_rank

N = 32 # 16
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
X = np.c_[np.cos(theta), np.sin(theta)]

## Complete complex 
st = sx.simplicial_complex([[v] for v in np.arange(len(X))], form='tree')
st.insert(combinations(st.vertices, 2))
st.expand(k=2)

## Form an implicit flag filtration 
M = clique_mod.Cliqueser_flag(len(X), 2)
M.init(pdist(X))

K = sx.rips_filtration(X, p = 2, form="rank")
K.order = 'reverse colex'

D1 = sx.boundary_matrix(K, p=1)
D2 = sx.boundary_matrix(K, p=2)

## Get edges and triangles
er = np.array(comb_to_rank(sx.faces(K,1), order='colex', n=sx.card(K,0)))
tr = np.array(comb_to_rank(sx.faces(K,2), order='colex', n=sx.card(K,0)))

## Ensure we can deflate the nullspace using apparent pairs
ap_co_1 = np.array([M.apparent_zero_cofacet(r, 1) for r in er])
# ap_co_1 = np.array([M.apparent_zero_facet(r, 2) for r in tr])
ap_co_2 = np.array([M.apparent_zero_cofacet(r, 2) for r in tr])

nullity_edg = np.sum(ap_co_1 != -1)
nullity_tri = np.sum(ap_co_2 != -1)
print(f"Can remove {nullity_edg}/{sx.card(K,1)} edges from K ({(nullity_edg/sx.card(K,1)) * 100:.1f}% reduction)")
print(f"Can remove {nullity_tri}/{sx.card(K,2)} triangles from K ({(nullity_tri/sx.card(K,2)) * 100:.1f}% reduction)")

## Rank computation short-cut for D2 
pivot_cols_D2 = ap_co_2 != -1
pivot_rows_D2 = ap_co_1 != -1


# %% 
from scipy.sparse.linalg import eigsh
from spirit.apparent_pairs import SpectralRI, deflate_sparse, UpLaplacian

## Initialize 
L = SpectralRI(K, p=2)

## Set weights using the filter function
diam_f = sx.flag_filter(pdist(X))
L._weights[0] = np.repeat(1e-8, sx.card(K,0))
L._weights[1] = diam_f(sx.faces(K,1))
L._weights[2] = diam_f(sx.faces(K,2))

i, j, p = 1.0, 1.5, 1
L._weights[1] = np.where(L._weights[1] > i, L._weights[1], 0)
L._weights[2] = np.where(L._weights[2] <= j, L._weights[2], 0)
print(sum(L._weights[1] != 0))
print(sum(np.where(np.logical_and(L._weights[p] > i, L._weights[p] <= j), L._weights[p], 0) != 0))

ri, ci = L._D[2].row, L._D[2].col
L._D[2].data[L._weights[1][ri] == 0] = 0
L._D[2].data[L._weights[2][ci] == 0] = 0

deflate_sparse(L._D[2])
sum(L.weights[1] != 0)
sum(L.weights[2] != 0)

## The above should match this
L.lower_left(i = i, j = j, p = 2)

## Deflate zeroes
D2_rank = np.linalg.matrix_rank(L._D[2].todense())
D2_ij = deflate_sparse(L._D[2])
print(f"deflated rank IJ: {np.linalg.matrix_rank(D2_ij.todense())}, shape = {D2_ij.shape}, (full rank = {D2_rank})")

## Detect apparent pairs and strip positive simplices from the nullspace
L.detect_pivots(pdist(X), p=2, f_type = "flag")
L._status[2]

L._D[2].data = np.where(L._status[2][L._D[2].col] > 0, 0, L._D[2].data)
D2_ij_ap = deflate_sparse(L._D[2])
print(f"rank IJ (ap): {np.linalg.matrix_rank(D2_ij_ap.todense())}, shape = {D2_ij_ap.shape}, (full rank = {D2_rank})")

print(f"dim/nnz full: {L._D[2].shape}, {L._D[2].nnz}")
print(f"dim/nnz reduced ij: {D2_ij.shape}, {D2_ij.nnz} ({100 - (D2_ij.nnz/L._D[2].nnz)*100:.0f}% reduction)")
print(f"dim/nnz reduced ap: {D2_ij_ap.shape}, {D2_ij_ap.nnz} ({100 - (D2_ij_ap.nnz/L._D[2].nnz)*100:.0f}% reduction)")


## SK-Sparse way of getting rank 
from scipy.sparse import csc_matrix, csc_array
from sksparse.cholmod import cholesky, cholesky_AAt
D2_ij_ap.data = D2_ij_ap.data.astype(np.float32)
D2_ij_ap = D2_ij_ap.tocsc()

F = cholesky_AAt(csc_matrix(D2_ij_ap))
np.sum(F.D() != 0)

cholesky(csc_array(np.eye(5)))

cholesky(L.lower_left(i = i, j = j, p = 2, deflate=True, apparent=True).as_sparse())

## Test 
L.detect_pivots(pdist(X), p = 2, f_type = "flag")
L.lower_left(i = i, j = j, p = 2, deflate=False).as_sparse()
L.lower_left(i = i, j = j, p = 2, deflate=True).as_sparse()
L.lower_left(i = i, j = j, p = 2, deflate=True, apparent=True).as_sparse()


LA = L.lower_left(i = i, j = j, p = 2, deflate=True, apparent=True)
LA.D = LA.D.tocsc()
F = cholesky(LA.as_sparse().tocsc())


# %% 
from bokeh.io import output_notebook
from bokeh.plotting import show
from pbsig.vis import figure_dgm
from pbsig.persistence import ph
output_notebook()

dgms = ph(K)
show(figure_dgm(dgms[1]))

## Select a quadrant
a,b,c,d = 0.10, 0.25, 1.5, 1.9
ranks = [0]*4

L = SpectralRI(K, p = 2)
L._weights[0] = np.repeat(1e-8, sx.card(K,0))
L._weights[1] = diam_f(sx.faces(K,1))
L._weights[2] = diam_f(sx.faces(K,2))
for cc, (i,j) in enumerate([(b,c), (a,c), (b,d), (a,d)]):
  L1 = L.lower_left(i,j,p=2)
  ranks[cc] = np.linalg.matrix_rank(L1 @ np.eye(L1.shape[0]))
  print(f"{L.D[2].shape}")
  print(f"rank: {ranks[cc]}")

(ranks[0] + ranks[3]) - (ranks[1] + ranks[2])

# %% Verify *all* the way back to the one that worked!
from pbsig.persistence import persistent_betti
L = SpectralRI(K, p=2)
a,b,c,d = 0.10, 0.25, 1.5, 1.9

mult, ranks = 0, [0]*4
for cc, (i,j) in enumerate([(b,c), (a,c), (b,d), (a,d)]):
  ii = np.searchsorted(diam_f(sx.faces(K,1)), i)
  jj = np.searchsorted(diam_f(sx.faces(K,2)), j)
  pb = persistent_betti(L._D[1].tolil(), L._D[2].tolil(), ii+1, jj, summands=True)
  print(f"PB: {pb[0]-pb[1]-pb[2]+pb[3]}, {pb}")
  mult += pb[3] if cc == 0 or cc == 3 else -pb[3]
# 224, 225, 352, 354
# 223 - 224 - 351 + 353

sum(np.where(diam_f(sx.faces(K,2)) <= j, 1, 0))

print("TRUTH")
L = SpectralRI(K, p=2)
for cc, (i,j) in enumerate([(b,c), (a,c), (b,d), (a,d)]):
  D2 = L._D[2].tolil()
  ii = diam_f(sx.faces(K,1)) > i
  jj = diam_f(sx.faces(K,2)) <= j
  print(np.linalg.matrix_rank(D2[ii,:][:,jj].todense()))
  print(f"# edges: {np.sum(ii)}, # triangles: {np.sum(jj)}")
# 1 = 224 - 224 - 352 + 353 # (right)

print("TEST")
## This has the wrong number of edges + triangles, but seems to have the correct pattern!
L = SpectralRI(K, p=2)
for cc, (i,j) in enumerate([(b,c), (a,c), (b,d), (a,d)]):
  L.reset()
  L._weights[0] = np.repeat(1e-8, sx.card(K,0))
  L._weights[1] = diam_f(sx.faces(K,1))
  L._weights[2] = diam_f(sx.faces(K,2))
  L.lower_left(i = i, j = j, p = 1)
  # print(np.linalg.matrix_rank(L._D[2].todense()))
  print(np.linalg.matrix_rank(L.D[2].todense()))
  # print(np.linalg.matrix_rank(L.D[2].todense()))
  print(f"# edges: {np.sum(L.weights[1] > 0)}, # triangles: {np.sum(L.weights[2] > 0)}")
  # Right: 224 - 224 - 352 - 353

def rank_cholmod(A):
  from scipy.sparse import csc_matrix, csc_array
  from sksparse.cholmod import cholesky, cholesky_AAt
  F = cholesky_AAt(A, beta=1e-6)
  return np.sum(F.D() > 1e-2)

print("Accelerated")
L = SpectralRI(K, p=2)
L._weights[0] = np.repeat(1e-8, sx.card(K,0))
L._weights[1] = diam_f(sx.faces(K,1))
L._weights[2] = diam_f(sx.faces(K,2))
for cc, (i,j) in enumerate([(b,c), (a,c), (b,d), (a,d)]):
  L1 = L.lower_left(i = i, j = j, p = 2)
  L.rank(i=i,j=j,p=2)
  np.linalg.matrix_rank(L1 @ np.eye(L1.shape[0]))
  #print(rank_cholmod(L.D[2].tocsc()))

from primate.functional import numrank


rank_cholmod(L1.D.tocsc())

## Rank 
L.rank(i,j,p=2)


import timeit
from primate.functional import numrank
from primate.diagonalize import lanczos
D2 = L1.D.tocsc().astype(np.float32)
timeit.timeit(lambda: numrank(D2 @ D2.T, seed=1), number = 30) / 30

numrank(D2 @ D2.T, seed=26, maxiter=10, deg=300, orth=25, ncv=30, verbose=True, num_threads=1)

# eigsh(D2 @ D2.T, k=1, which='SM')
hutch(D2 @ D2.T, fun="smoothstep", a=0, b=gap, atol=0.45, maxiter=1500, deg=5, orth=0, ncv=5, verbose=True)
np.linalg.matrix_rank((D2 @ D2.T) @ np.eye(D2.shape[0]))


## 
L.rank(i,j,p=2)


np.sum(np.linalg.svd(L.D[2].todense())[1] > 1)

from pbsig.betti import betti_query, BettiQuery, SpectralRankInvariant
# betti_query(K, f = f, p = 1, i = i, j = j)
BQ = BettiQuery(K, p = 1)
BQ.weights[0] = np.repeat(1e-8, sx.card(K,0))
BQ.weights[1] = diam_f(sx.faces(K,1))
BQ.weights[2] = diam_f(sx.faces(K,2))
BQ(a,b,c,d)


SI = SpectralRankInvariant(K, [diam_f])
SI.sieve = [[a,b,c,d]]
SI.sift(p = 1)

BQ.operator(0, i=a,j=b,k=c,l=d)
BQ.operator(1, i=a,j=b,k=c,l=d)
BQ.operator(2, i=a,j=b,k=c,l=d)
BQ.operator(3, i=a,j=b,k=c,l=d)


# np.linalg.matrix_rank(L._D[2].todense())


## Test and make sure the apparent pairs we detected in the nullspace are indeed so
from pbsig.persistence import reduction_pHcol, low_entry
D1, D2 = L._D[1].tolil(), L._D[2].tolil()
R1, R2, V1, V2 = reduction_pHcol(D1, D2)
in_nullspace_test = L._status[2] > 0 
in_nullspace_true = low_entry(R2) == -1
assert np.any(np.logical_and(in_nullspace_test, ~in_nullspace_true)) == False








# L._D[1].data
# L.compressed_D(1)

# L.lower_left(i = 1.0, j = 1.5, p = 2) # L0.reset()
# L.lower_left(i = 1.0, j = 1.5, p=1)


# ## Detect apparent pairs
# L0.detect_pivots(pdist(X), f_type = "flag")
# L0._p_status

# L0.rank(i = 1.0, j = 1.5)

from primate.functional import numrank
numrank(L0)
np.linalg.matrix_rank(L0 @ np.eye(32))


L0.rank(i=1.0, j=1.5)

nz_pr = L0.pr[np.unique(L0._D.row[L0._D.data != 0])]
nz_qr = L0.pr[np.unique(L0._D.col[L0._D.data != 0])]


L0.lower_left(i = 1.0, j = 1.5)


## Retrieve the boundary matrix directly
L0.compressed()




# %% Get the rips persistence
from pbsig.persistence import ph
from pbsig.vis import figure_dgm, show
from bokeh.io import output_notebook
output_notebook()
K = sx.rips_filtration(pdist(X), p=2)
dgms = ph(K)
show(figure_dgm(dgms[1]))


# %% 
from ripser import ripser
dgms = ripser(X, maxdim=1)


# ## Update the fitted boundary matrix
# self._D = deflate_sparse(self._D)
# self.shape = self._D.shape
# def reweight_and_compress(self, q_weights = None, p_weights = None, apparent: bool = False):
#   """Updates the weights of the complex and deflates the nullspace. 
  
#   This function may modify the .shape and .D attributes. 
#   """
#   self.p_weights = np.ones(len(self.p_weights)) if p_weights is None else np.array(p_weights)
#   self.q_weights = np.ones(len(self.q_weights)) if q_weights is None else np.array(q_weights)
  
#   ## Remove the 
  

  
from scipy.sparse.linalg import eigsh
np.sort(eigsh(L0, k=L0.shape[0]-1, which='LM', return_eigenvectors=False))
np.sort(np.linalg.svd(L0.D.todense())[1]**2)
