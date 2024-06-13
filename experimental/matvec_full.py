import numpy as np 
from combin import comb_to_rank, rank_to_comb
from itertools import cycle, islice, product, combinations
from math import comb
from itertools import combinations
from spirit.apparent_pairs import clique_mod
from simplextree import SimplexTree
from spirit.apparent_pairs import boundary_matrix, clique_mod

# triangles = list(combinations([0,1,2,3,4], 3))
# [list(combinations(t, 2)) for t in triangles]
# [list(combinations(t,2)) for t in combinations([0,1,2,3,4], 3)]

np.random.seed(1234)
n, k = 10, 3
# x = np.ones(comb(n, k-1))
x = np.arange(comb(n, k-1))
# x = np.random.uniform(size=comb(n, k-1))

## Global algorithm - launch thread for each k-combination of an n-simplex 
n_simplex = np.arange(n) 
# deg_map = { i : set({}) for i in range(comb(len(n_simplex), k-1)) }
deg_map = { i : 0 for i in range(comb(len(n_simplex), k-1)) }
# # for k_simplex in combinations(n_simplex, k):
for r in range(comb(n, k)):
  # for p_simplex in combinations(k_simplex, k-1): # only k of these 
  # r = comb_to_rank(p_simplex, order='colex', n=len(n_simplex))
  p_simplices = clique_mod.enum_boundary(r, k-1, n)
  for i in p_simplices:
    deg_map[i] += 1 
  # for (i,j) in combinations(p_simplices, 2):
  #   deg_map[i] |= set([r])
  #   deg_map[j] |= set([r])
    # deg_map[i] += 1
# deg = np.array([len(deg_map[r]) for r in range(comb(n, k-1))])
deg = np.array([deg_map[r] for r in range(comb(n, k-1))])

# st = SimplexTree([n_simplex])
# true_deg = np.array([sum([1 for c in st.cofaces(s) if len(c) == k]) for s in st.simplices(k-2)])
# assert np.allclose(deg, true_deg)
y = x * deg 
sgn_pattern = np.where(np.arange(k) % 2 == 0, 1, -1)
sgn_pattern = np.array([si*sj for si,sj in list(combinations(sgn_pattern, 2))])
# for k_simplex in combinations(n_simplex, k):
for r in range(comb(n, k)):
  p_simplices = clique_mod.enum_boundary(r, k-1, n)
  assert np.all(np.argsort(p_simplices) == np.arange(k))
  for (i, j), s in zip(combinations(p_simplices, 2), sgn_pattern): # only k of these 
    y[i] += s * x[j]
    y[j] += s * x[i]
print(y)
# assert np.allclose(y, 0)

## Check against implementation
cofacets = np.sort(np.array(comb_to_rank(combinations(n_simplex, k), order='colex', n=n)))
facets = np.sort(np.array(comb_to_rank(combinations(n_simplex, k-1), order='colex', n=n)))
D = boundary_matrix(k-1, cofacets, facets, n=n)
assert np.allclose((D @ D.T).diagonal(), deg)
assert np.allclose(y, (D @ D.T) @ x)

# k_simplex = rank_to_comb(cofacets, k=k, n=n, order='colex')[0]

## Loop unrolling for various k 
import string
K = 3
sgn_pattern = np.where(np.arange(K) % 2 == 0, 1, -1)
sgn_pattern = np.array([si*sj for si,sj in list(combinations(sgn_pattern, 2))])

## Programmatically create the matvec
weighted = False
st = { f'y[{i}] += ' : '' for i in string.ascii_lowercase[:K] }
for (i,j),s in zip(combinations(string.ascii_lowercase[:K], 2), sgn_pattern):
  yi, yj = f'y[{i}] += ', f'y[{j}] += '
  if not weighted: 
    st[yi] += f"({s}) * x[{j}] + "
    st[yj] += f"({s}) * x[{i}] + "
  else:
    st[yi] += f"({s}) * x[{j}] * wf[{i}] * ws[{0}] * wf[{j}] + "
    st[yj] += f"({s}) * x[{i}] * wf[{j}] * ws[{0}] * wf[{i}] + "
print('\n'.join([key + val for key, val in st.items()]))
# statements = np.ravel([[f'y[{i}] += ({s}) * x[{j}]', f'y[{j}] += ({s}) * x[{i}]'] for (i,j),s in zip(combinations(string.ascii_lowercase[:K], 2), sgn_pattern)])
# print('\n'.join(statements))

y = x * deg 
for r in range(comb(n, k)):
  p_simplices = clique_mod.enum_boundary(r, k-1, n)
  a,b,c,d = p_simplices
  y[a] += x[c] - (x[b] + x[d])
  y[b] += x[d] - (x[a] + x[c])
  y[c] += x[a] - (x[b] + x[d])
  y[d] += x[b] - (x[a] + x[c])

## K = 5 
for r in range(comb(n, k)):
  p_simplices = clique_mod.enum_boundary(r, k-1, n)
  a, b, c, d, e = p_simplices
  y[a] += (x[c] + x[e]) - (x[b] + x[d])
  y[b] += x[d] - (x[a] + x[c] + x[e])
  y[c] += (x[a] + x[e]) - (x[b] + x[d])
  y[d] += x[b] - (x[a] + x[c] + x[e])
  y[e] += (x[a] + x[c]) - (x[b] + x[d])


# K = 5 
# for (i, j), s in zip(combinations(p_simplices, 2), sgn_pattern): # only k of these 
# assert comb(K,2) == len(sgn_pattern)
# a,b,c,d,e = 0,1,2,3,4

## K = 6
# y[a] += x[c] + x[e] - (x[b] + x[d] + x[f])
# y[b] += x[d] + x[f] - (x[a] + x[c] + x[e])
# y[c] += x[a] + x[e] - (x[b] + x[d] + x[f])
# y[d] += x[b] + x[f] - (x[a] + x[c] + x[e])
# y[e] += x[a] + x[c] - (x[b] + x[d] + x[f])
# y[f] += x[b] + x[d] - (x[a] + x[c] + x[e]) 


## K = 7 
y[a] += x[c] + x[e] + x[g] - (x[b] + x[d] + x[f])
y[b] += x[d] + x[f] - (x[a] + x[c] + x[e] + x[g])
y[c] += x[a] + x[e] + x[g] - (x[b] + x[d] + x[f])
y[d] += x[b] + x[f] - (x[a] + x[c] + x[e] + x[g])
y[e] += x[a] + x[c] + x[g] - (x[b] + x[d] + x[f])
y[f] += x[b] + x[d] - (x[a] + x[c] + x[e] + x[g])
y[g] += x[a] + x[c] + x[e] - (x[b] + x[d] + x[f])

cofacets = np.sort(np.array(comb_to_rank(combinations(n_simplex, k), order='colex', n=n)))
facets = np.sort(np.array(comb_to_rank(combinations(n_simplex, k-1), order='colex', n=n)))

# rank_to_comb(cofacets, k=k, n=n, order='colex')
# rank_to_comb(facets, k=k-1, n=n, order='colex')
# clique_mod.build_coo(4, 1, cofacets, facets)
# d, (ri,ci) = clique_mod.build_coo(4, 1, cofacets, facets)
D = boundary_matrix(k-1, cofacets, facets, n=n)
D.todense()
(D @ D.T) 
(D @ D.T) @ x

cofacets_r = np.flip(np.sort(cofacets)).copy()
facets_r = np.flip(np.sort(facets)).copy()

# rank_to_comb(cofacets_r, k=k, n=n, order='colex')
# rank_to_comb(facets_r, k=k-1, n=n, order='colex')

D = boundary_matrix(k-1, cofacets_r, facets_r, n=n).todense()
(D @ D.T) @ x 

import splex as sx
np.random.seed(1234)
st = SimplexTree([n_simplex])
D = sx.boundary_matrix(st, p=k-1)
# x = np.random.uniform(size=D.shape[0])
D @ D.T @ x



### very low memory n-simplex matvec
class UpLaplacian():
  def __init__(self, n: int, k: int):
    self.n = n
    self.k = k
    self.deg = np.zeros(comb(n, k-1), dtype=np.float32)
  
  def precompute_deg(self):
    self.deg = np.zeros(comb(self.n, self.k-1), dtype=np.float32)
    for r in range(comb(self.n, self.k)):
      p_simplices = clique_mod.enum_boundary(r, self.k-1, self.n)
      for i in p_simplices:
        self.deg[i] += 1 

  def _matvec(self, x: np.ndarray):
    if self.k == 3:
      y = x * self.deg
      for r in range(comb(self.n, 3)):
        i,j,q = clique_mod.enum_boundary(r, self.k-1, self.n)
        y[i] -= x[j]
        y[j] -= x[i]
        y[i] += x[q]
        y[q] += x[i]
        y[j] -= x[q]
        y[q] -= x[j]

def up_laplacian_matvec_2():
  x = np.ones(comb(n, 2)) #np.random.uniform(size=comb(n, k-1))
  y = x * deg
  # sgn_pattern = np.array([-1,+1,-1])
  for r in range(comb(n, 3)):
    i,j,q = clique_mod.enum_boundary(r, k-1, n)
    y[i] -= x[j]
    y[j] -= x[i]
    y[i] += x[q]
    y[q] += x[i]
    y[j] -= x[q]
    y[q] -= x[j]



## Sparse (k-2)-UpLaplacian over 'n' vertices enumerating over 'S' with faces 'F'
class UpLaplacian:
  def __init__(self, S: np.ndarray, F: np.ndarray, n: int, k: int):
    assert np.all(np.argsort(F) == np.arange(len(F))), "Faces array 'F' must be ordered."
    self.n = n
    self.k = k
    self.deg = np.zeros(len(F), dtype=np.float32)
    self.S = S 
    self.F = F

  def precompute_deg(self):
    self.deg = np.zeros(len(self.F), dtype=np.float32)
    for s in self.S:
      p_simplices = clique_mod.enum_boundary(s, self.k-1, self.n)
      for i in p_simplices:
        self.deg[i] += 1 

  def _matvec(self, x: np.ndarray, y: np.ndarray):
    np.multiply(x, self.deg, y) # y = x * deg
    ps = np.zeros(self.k, dtype=np.int64)
    for s in self.S:
      ps = clique_mod.enum_boundary(s, self.k-1, self.n)
      if self.k == 2:
        a,b = np.searchsorted(self.F, ps)
        y[a] -= x[b]
        y[b] -= x[a]
      elif self.k == 3:
        a,b,c = np.searchsorted(self.F, ps)
        y[a] += (x[c] - x[b])
        y[b] -= (x[b] + x[c])
        y[c] += (x[a] - x[b])
      elif self.k == 4:
        a,b,c,d = np.searchsorted(self.F, ps)
        y[a] += x[c] - (x[b] + x[d])
        y[b] += x[d] - (x[a] + x[c])
        y[c] += x[a] - (x[b] + x[d])
        y[d] += x[b] - (x[a] + x[c])



from itertools import combinations
from combin import comb_to_rank
np.random.seed(1234)
n, k = 8, 3
BT = np.array([[comb(ni,ki) for ni in range(n)] for ki in range(k+2)]).astype(np.int64)
S = comb_to_rank(np.array(list(combinations(range(n),k))), n = n, order='colex')
F = np.sort(comb_to_rank(np.array(list(combinations(range(n),k-1))), n = n, order='colex'))

x = np.arange(len(F))
y = np.zeros(len(F))
L = UpLaplacian(S, F, n, k)
L.precompute_deg()
L._matvec(x, y)
print(y)

ps = clique_mod.enum_boundary(0, k - 1, n)
np.searchsorted(F, ps)

x = np.arange(len(F))
D = boundary_matrix(k-1, S, F, n=n)
D @ D.T @ x


def searchsorted(a: np.ndarray, v: np.ndarray):
  """ Searches for index locations of all values 'v' in 'a' via binary search, storing results in 'y' """
  n_bins = a.size
  left: int = 0
  right: int = n_bins-1
  out = np.zeros(v.size, dtype=np.int64)
  for i, x in enumerate(v): 
    while left < right:
      m: int = left + (right - left) // 2
      if a[m] < x:
        left = m + 1
      else:
        right = m
    out[i] = right
    left = right 
    right = n_bins - 1
  return out


searchsorted(F, np.array([10, 14, 17]))

a = np.sort(np.random.choice(range(100), size=26, replace=False))
searchsorted(a, np.array([31, 52, 59, 81]))
np.flatnonzero(a == 81)