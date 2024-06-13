# %% Imports 
from typing import Callable
import numpy as np 
import numba as nb 
from combin import rank_to_comb, comb_to_rank
from itertools import combinations
from math import comb

do_bounds_check = True
n, k = 7, 3
N, M = comb(n,k-1), comb(n,k)
BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+6)]).astype(np.int64)

# %% JIT 
@nb.jit(nopython=True, boundscheck=do_bounds_check)
def get_max(top: int, bottom: int, pred: Callable, *args):
  if ~pred(bottom, *args):
    return bottom
  size = (top - bottom)
  while (size > 0):
    step = size >> 1
    mid = top - step
    if ~pred(mid, *args):
      top = mid - 1
      size -= step + 1
    else:
      size = step
  return top

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def find_k_pred(w: int, r: int, m: int, BT: np.ndarray) -> bool:
  # print(f"C({w},{m})")
  return BT[m][w] <= r

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def get_max_vertex(r: int, m: int, n: int, BT: np.ndarray) -> int:
  k_lb: int = m - 1
  return 1 + get_max(n, k_lb, find_k_pred, r, m, BT)

@nb.jit(nopython=True,boundscheck=do_bounds_check)
def k_boundary_cpu(n: int, simplex: int, dim: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int = simplex
  idx_above: int = 0
  j = n - 1
  for k in np.flip(np.arange(dim+1)):
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1
    c = BT[k+1][j]
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += BT[k][j]
    out[dim-k] = face_index

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def precompute_deg(n: int, k: int, N: int, M: int, BT: np.ndarray) -> np.ndarray:
  deg = np.zeros(N)
  k_faces = np.zeros(k, dtype=np.int32)
  for r in range(M):
    k_boundary_cpu(n, simplex=r, dim=k-1, BT=BT, out=k_faces)
    deg[k_faces] += 1
  return deg

# %% New functions
from math import floor, sqrt

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def rank_to_comb_lex2(r: int, n: int) -> tuple:
  i: int = int(n - 2 - floor(sqrt(-8*r + 4*n*(n-1)-7)/2.0 - 0.5))
  j: int = int(r + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return (i,j)

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def comb_to_rank_lex2(i: int, j: int, n: int) -> int:
  i, j = (j, i) if j < i else (i, j)
  return int(n*i - i*(i+1)/2 + j - i - 1)

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def rank_to_comb_colex(simplex: int, n: int, k: int, BT: np.ndarray, out: np.ndarray):
  K: int = n - 1
  for ki in range(1, k):
    m = k - ki + 1
    K = get_max_vertex(simplex, m, n, BT)
    # assert comb(K-1,m) <= simplex and simplex < comb(K, m)
    out[ki-1] = K-1
    simplex -= BT[m][K-1]
  out[-1] = simplex

# %% 

# simplex = 13
# k_faces = np.zeros(k)
# rank_to_comb_colex(13, n=7, k=3, BT=BT, out=k_faces)

# rank_to_comb_colex(0, n=7, k=3, BT=BT, out=k_faces)
# print(k_faces)

# rank_to_comb_colex(20, n=7, k=3, BT=BT, out=k_faces)


# %% 
@nb.jit(nopython=True, boundscheck=do_bounds_check)
def simplex_weight(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
  K = dim + 1
  labels = np.zeros(K)
  rank_to_comb_colex(simplex, n=n, k=K, BT=BT, out=labels)
  s_weight = 0.0
  for i in range(K):
    for j in range(i+1, K):
      s_weight = max(s_weight, weights[comb_to_rank_lex2(labels[i], labels[j], n)])
  return s_weight

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def k_coboundary(simplex: int, n: int, dim: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int = simplex
  idx_above: int = 0 
  j: int = n - 1
  k: int = dim + 1
  c: int = 0
  while j >= k:
    while BT[k][j] <= idx_below:
      idx_below -= BT[k][j]
      idx_above += BT[k+1][j]
      j -= 1
      k -= 1
      # assert k != -1, "Coboundary enumeration failed"
    cofacet_index = idx_above + BT[k+1][j] + idx_below
    # print(f"{cofacet_index} = {idx_above} + {BT[k+1][j]} + {idx_below} (j={j},k={k})")
    j -= 1
    out[c] = cofacet_index
    c += 1

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def k_boundary(simplex: int, n: int, dim: int, BT: np.ndarray, out: np.ndarray):
  idx_below: int = simplex
  idx_above: int = 0 
  j: int = n - 1
  for ki in range(dim+1):
    k = dim - ki
    j = get_max_vertex(idx_below, k + 1, j, BT) - 1
    c = BT[k+1][j]
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += BT[k][j]
    out[ki] = face_index

# %% 
@nb.jit(nopython=True, boundscheck=do_bounds_check)
def zero_cofacet(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int:
  '''Given a dim-dimensional simplex, find its lexicographically maximal cofacet with identical simplex weight'''
  c_weight: float = simplex_weight(simplex, dim=dim, n=n, weights=weights, BT=BT)
  zero_cofacet: int = -1
  k_cofacets = np.zeros(n-(dim+1), dtype=np.int64)
  k_coboundary(simplex, n=n, dim=k-1, BT=BT, out=k_cofacets)
  for c_cofacet in k_cofacets:
    cofacet_weight = simplex_weight(c_cofacet, dim=dim+1, n=n, weights=weights, BT=BT)
    if cofacet_weight == c_weight:
      zero_cofacet = c_cofacet
      break
  return zero_cofacet

@nb.jit(nopython=True, boundscheck=do_bounds_check)
def zero_facet(simplex: int, dim: int, n: int, weights: np.ndarray, BT: np.ndarray) -> int:
  '''Given a dim-dimensional simplex, find its lexicographically minimal facet with identical simplex weight'''
  c_weight: float = simplex_weight(simplex, dim=dim, n=n, weights=weights, BT=BT)
  zero_facet: int = -1
  k_facets = np.zeros(dim+1, dtype=np.int64)
  k_boundary(simplex, n=n, dim=dim, BT=BT, out=k_facets)
  for c_facet in k_facets:
    facet_weight = simplex_weight(c_facet, dim=dim-1, n=n, weights=weights, BT=BT)
    # print(f"{c_facet} => {facet_weight:.5f}")
    if facet_weight == c_weight:
      zero_facet = c_facet
      break
  return zero_facet

# %% 
@nb.jit(nopython=True, boundscheck=do_bounds_check)
def construct_flag_dense(N: int, dim: int, n: int, eps: float, apparent: bool, weights: np.ndarray, BT: np.ndarray):
  """Constructs d-simplices of a dense flag complex up to 'eps', optionally discarding apparent pairs."""
  S = []
  if apparent:
    ## Normally we would find AP's; here we only keep non-apparent d-simplices 
    for r in range(N): 
      w = simplex_weight(r, dim=dim, n=n, weights=weights, BT=BT)
      if w <= eps: 
        c = zero_cofacet(r, dim=dim, n=n, weights=weights, BT=BT)
        z = zero_facet(c, dim=dim+1, n=n, weights=weights, BT=BT)
        if r == 11: 
          print(f"Debug: {r} (c: {c}, z: {z})")
        if c == -1 or z != r:
          print(f"Keeping: {r} (c: {c}, z: {z})")
          S.append(r)
        else:
          S.append(-1)
  else: 
    for r in range(N):
      w = simplex_weight(r, dim=dim, n=n, weights=weights, BT=BT)
      if w <= eps:
        S.append(r)
  return np.array(S)

# c = zero_cofacet(r, dim=k-1, n=n, weights=weights, BT=BT)
# if c != -1 and zero_facet(c, dim=k, n=n, weights=weights, BT=BT) == r:
# ap_pairs = []
# for ii, r in enumerate(range(N)):
# c = zero_cofacet(r, dim=k-1, n=n, weights=weights, BT=BT)
# if c != -1 and zero_facet(c, dim=k, n=n, weights=weights, BT=BT) == r:
#   ap_pairs.append((r,c))


# @nb.jit(nopython=True, boundscheck=do_bounds_check)
# def apparent_pairs(S: np.ndarray, dim: int, n: int, weights: np.ndarray, BT: np.ndarray):
#   ap_pairs = []
#   for ii, r in enumerate(S):
#     c = zero_cofacet(r, dim=k-1, n=n, weights=weights, BT=BT)
#     if c != -1 and zero_facet(c, dim=k, n=n, weights=weights, BT=BT) == r:
#       ap_pairs.append((r,c))

# %% Construct 
from spirit.apparent_pairs import SpectralRI
import splex as sx
from scipy.spatial.distance import pdist, cdist, squareform
from pbsig.persistence import sw_parameters

# for seed in range(120):
n, k = 16, 3
np.random.seed(0)
X = np.random.uniform(size=(n, 2))
dX = pdist(X)
BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+6)]).astype(np.int64)
S = construct_flag_dense(comb(n,k), k-1, n, np.inf, True, dX, BT)
print(S)
print(S[S > -1])
print(np.sum(S > -1))
# print(f"seed: {seed} => {len(S)}")


RI = SpectralRI(n=len(X), max_dim=4)
RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag", threshold=np.inf)
RI.construct(dX, p=2, apparent=True, discard=False, filter="flag", threshold=np.inf)
RI._simplices[2]
np.flip(RI._simplices[2][RI._status[2] <= 0])

# simplex_weight(2, k-1, n, dX, BT)

N = 700 
M = 23
f = lambda t: np.cos(t) + np.cos(3*t)
SW = sliding_window(f, bounds=(0, 12*np.pi))
_, tau_opt = sw_parameters(bounds=(0, 12*np.pi), d=24, L=6)
X = SW(n=N, d=M, tau=tau_opt)
dX = pdist(X)
n = len(X)

ER = sx.enclosing_radius(dX)
RI = SpectralRI(n=len(X), max_dim=4)
RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=True, discard=False, filter="flag", threshold=np.inf)
RI.construct(dX, p=2, apparent=True, discard=False, filter="flag", threshold=np.inf)
# print(f"Fraction non-AP: {(len(RI._simplices[2]) / comb(len(X), 3))*100:.3f}%")
print(f"Fraction apparent: {((np.sum(RI._status[2] > 0) / len(RI._simplices[2]))) * 100:.2f}%")
# RI._simplices[2]

RI._simplices[2][RI._status[2] > 0]

## Apparent pairs calculation
BT = np.array([[comb(ni, ki) for ni in range(n+1)] for ki in range(k+6)]).astype(np.int64)

## IS very different 
ap_pairs = []
for ii, r in enumerate(range(comb(n, k))):
  c = zero_cofacet(r, dim=k-1, n=n, weights=dX, BT=BT)
  if c != -1 and zero_facet(c, dim=k, n=n, weights=dX, BT=BT) == r:
    ap_pairs.append((r,c))
  if ii % 10000 == 0:
    print(ii)

len(ap_pairs)
# 56652516 / len(RI._simplices[2])






k_faces = np.zeros(k+1, dtype=np.int64)
k_boundary(c, n, dim=k, BT=BT, out=k_faces)

c_weight = simplex_weight(c, dim=k, n=n, weights=dX, BT=BT) 
k_face_weights = np.array([simplex_weight(f, dim=k-1, n=n, weights=dX, BT=BT) for f in k_faces])

k_cofaces = np.zeros(n-(k+1), dtype=np.int64)
k_coboundary(r, n, dim=k, BT=BT, out=k_faces)

np.array([simplex_weight(f, dim=k-1, n=n, weights=dX, BT=BT) for f in k_faces])

## Test boundary 
k_faces = np.zeros(k, dtype=np.int64)
for r in range(comb(n,k)):
  k_boundary(r, n, k-1, BT, k_faces)
  print(f"{r} => {k_faces}")
  assert np.all(k_faces <= comb(n, k-1))
  
  B = list(sx.Simplex(rank_to_comb(r, n=n, k=k, order='colex')).boundary())
  k_faces_true = comb_to_rank(B, k=k-1, n=n, order='colex')
  assert np.all(k_faces == k_faces_true)



N = 700
M = 23
X = SW(n=N, d=M, tau=tau_opt)
dX = pdist(X)
DX = squareform(dX)
ER = sx.enclosing_radius(DX)


k_faces = np.zeros(n-k, dtype=np.int64)
k_coboundary(13, n=n, dim=k-1, BT=BT, out=k_faces)
print(k_faces)

# %% 




# %%
s = rank_to_comb(0, k=k, order='colex', n=n)
[comb_to_rank(np.sort(np.append(s, [xj])), order='colex', n=n) for xj in np.setdiff1d(np.arange(n), s)]

from spirit.apparent_pairs import clique_mod
clique_mod.enum_coboundary(13, k-1, n, True)




for r in range(comb(n, k)):
  rank_to_comb_colex(r, n=7, k=3, BT=BT, out=k_faces)
  cc = np.flip(rank_to_comb(r, k=3, order='colex', n=n))
  assert np.all(cc == k_faces)
  print(f"{r} => {k_faces}")