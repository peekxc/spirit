from typing import Callable, Optional, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb
from itertools import combinations
from combin import rank_to_comb, comb_to_rank
from scipy.sparse import coo_array, sparray, coo_matrix
from scipy.sparse.linalg import LinearOperator
from typing import * 

import splex as sx
import _ripser as rip_mod
import _clique as clique_mod

## from: https://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r
def index_of(a: List[Hashable], b: List[Hashable], default: Any = None) -> List[int]:
  """Finds the index of each element A[i] in B """
  b_dict = {x: i for i, x in enumerate(b)}
  return np.array([b_dict.get(x, default) for x in a])

def deflate_sparse(A: sparray, mask: np.ndarray = None):
  """'Deflates' a given sparse array 'A' by removing it's rows and columns that are all zero.   
  Returns a new sparse matrix with the same data but potentially diffferent shape. 
  """
  from hirola import HashTable
  A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
  hr = HashTable(1.25*A.shape[0] + 16, A.row.dtype)
  hc = HashTable(1.25*A.shape[1] + 16, A.col.dtype)
  non_zero = A.data != 0 if mask is None else mask
  assert len(non_zero) == len(A.data), "Mask invalid! Must be length of data array."
  ri, ci = np.unique(A.row[non_zero]), np.unique(A.col[non_zero])
  hr.add(ri)
  hc.add(ci)
  ## Use matrix for sksparse
  A_deflated = coo_matrix((A.data[non_zero], (hr[A.row[non_zero]], hc[A.col[non_zero]])), shape=(hr.length, hc.length))
  # A_deflated = coo_array((A.data[non_zero], (hr[A.row[non_zero]], hc[A.col[non_zero]])), shape=(hr.length, hc.length))
  A_deflated.eliminate_zeros()
  return A_deflated # , np.unique(A.row), np.unique(A.col)

def _h0_apparent_pairs(K: sx.ComplexLike, f: Callable, refinement: str = "lex"):
  n = sx.card(K, 0)
  if sx.card(K,1) == 0: return []
  edges = np.array(list(sx.faces(K,1))).astype(np.uint16)
  E_ranks = comb_to_rank(edges, n=n, order='lex')

  ## Store the initial list of apparent pair candidates
  pair_candidates = []

  ## Since n >> k in almost all settings, start by getting apparent pair candidates from the p+1 simplices
  for e in E_ranks:
    i, j = rank_to_comb(e, k=2, n=n, order='lex')
    facets = [[i], [j]]
    facet_weights = f(facets)
    same_value = np.isclose(facet_weights, f([i,j]))
    if any(same_value):
      ## Choose the "youngest" facet, which is the *maximal* in lexicographical order
      lex_min_ind = int(np.flatnonzero(same_value)[-1])
      pair_candidates.append((facets[lex_min_ind], [i,j]))
    
  ## Now filter the existing pairs via scanning through each p-face's cofacets
  true_pairs = []
  for v,e in pair_candidates:
    facet_weight = f(v)
    
    ## Find the "oldest" cofacet, which is the *minimal* in lexicographical order
    max_cofacet = None
    for k in range(n): # reversed for maximal
      ## NOTE: equality is necessary here! Using <= w/ small rips filtration yields 16 pairs, whereas equality yields 48 pairs. 
      if sx.Simplex(k) != sx.Simplex(v) and np.isclose(facet_weight, f(sx.Simplex([v,k]))): 
        max_cofacet = sx.Simplex((k,v))
        break
    
    ## If the relation is symmetric, then the two form an apparent pair
    if max_cofacet is not None and max_cofacet == sx.Simplex(e):
      true_pairs.append((tuple([v]), max_cofacet))
  
  return true_pairs

def _h1_apparent_pairs(K: sx.ComplexLike, f: Callable, refinement: str = "lex", progess: bool = False):
  n = sx.card(K, 0)
  if sx.card(K,2) == 0: return []
  triangles = np.array(list(sx.faces(K,2))).astype(np.uint16)
  T_ranks = comb_to_rank(triangles, n=n, order='lex')

  ## Store the initial list of apparent pair candidates
  pair_candidates = []

  ## Since n >> k in almost all settings, start by getting apparent pair candidates from the p+1 simplices
  for t in T_ranks:
    i, j, k = rank_to_comb(t, k=3, n=n, order='lex')
    facets = [[i,j], [i,k], [j,k]]
    facet_weights = f(facets)
    same_value = np.isclose(facet_weights, f([i,j,k]))
    if any(same_value):
      ## Choose the "youngest" facet, which is the *maximal* in lexicographical order
      lex_min_ind = int(np.flatnonzero(same_value)[-1])
      pair_candidates.append((facets[lex_min_ind], [i,j,k]))
    
  ## Now filter the existing pairs via scanning through each p-face's cofacets
  true_pairs = []
  for e,t in pair_candidates:
    i,j = e
    facet_weight = f(e)
    
    ## Find the "oldest" cofacet, which is the *minimal* in lexicographical order
    max_cofacet = None
    for k in range(n): # reversed for maximal
      ## NOTE: equality is necessary here! Using <= w/ small rips filtration yields 16 pairs, whereas equality yields 48 pairs. 
      if k != i and k != j and np.isclose(facet_weight, f([i,j,k])): 
        max_cofacet = sx.Simplex((i,j,k))
        break
    
    ## If the relation is symmetric, then the two form an apparent pair
    if max_cofacet is not None and max_cofacet == sx.Simplex(t):
      true_pairs.append((tuple(e), max_cofacet))
  
  return true_pairs

def apparent_pairs(K: sx.ComplexLike, f: Callable, p: int = 0, refinement: str = "lex"):
  """Finds the H1 apparent pairs of lexicographically-refined clique filtration.

  A persistence pair (tau, sigma) is said to be *apparent* iff: 
    1. tau is the youngest facet of sigma 
    2. sigma is the oldest cofacet of tau 
    3. the pairing has persistence |f(sigma)-f(tau)| = 0 
  
  Parameters: 
    K: Simplicial complex.
    f: filter function defined on K.
    refinement: the choice of simplexwise refinement. Only 'lex' is supported for now. 

  Returns: 
    pairs (e,t) with zero-persistence in the H1 persistence diagram.

  Details: 
    Observe tau is the facet of sigma with the largest filtration value, i.e. f(tau) >= f(tau') for any tau' \\in facets(sigma)
    and sigma is cofacet of tau with the smallest filtration value, i.e. f(sigma) <= f(sigma') for any sigma' \\in cofacets(tau). 
    There are potentially several cofacets of a given tau, thus to ensure uniqueness, this function assumes the 
    filtration induced by f is a lexicographically-refined simplexwise filtrations. 
    
    Equivalently, for lexicographically-refined simplexwise filtrations, we have that a 
    zero-persistence pair (tau, sigma) is said to be *apparent* iff: 
      1. tau is the lexicographically *maximal* facet of sigma w/ f(tau) = f(sigma)
      2. sigma is the lexicographically *minimal* cofacet of sigma w/ f(sigma) = f(tau)

    Note that Bauer define similar notions but under the reverse colexicographical ordering, in which case the notions 
    of minimal and maximal are reversed.

    What is known about apparent pairs: 
      - Any apparent pair in a simplexwise filtration is a persistence pair, regardless of the choice of coefficients
      - Apparent pairs often form a large portion of the total number of persistence pairs in clique filtrations. 
      - Apparent pairs of a simplexwise filtrations form a discrete gradient in the sense of Discrete Morse theory 

    Moreover, if K is a Rips complex and all pairwise distances are distinct, it is knonw that the persistent pairs 
    w/ 0 persistence of K in dimension 1 are precisely the apparent pairs.
  """
  if p == 0: 
    return _h0_apparent_pairs(K,f,refinement)
  elif p == 1: 
    return _h1_apparent_pairs(K,f,refinement)
  else: 
    raise NotImplementedError("Haven't implemented higher AP calculations")

  # result = []
  # for T in K['triangles']:  
  #   T_facets = comb_to_rank(combinations(T, 2), k=2, n=len(K['vertices']), order="lex")
  #   max_facet = T_facets[np.max(np.flatnonzero(d[T_facets] == np.max(d[T_facets])))] # lexicographically maximal facet 
  #   n = len(K['vertices'])
  #   u, v = rank_to_comb(max_facet, k=2, n=n)
  #   same_diam = np.zeros(n, dtype=bool)
  #   for j in range(n):
  #     if j == u or j == v: 
  #       continue
  #     else: 
  #       cofacet = np.sort(np.array([u,v,j], dtype=int))
  #       cofacet_diam = np.max(np.array([d[comb_to_rank(face, k=2, n=n, order="lex")] for face in combinations(cofacet, 2)]))
  #       if cofacet_diam == d[max_facet]:
  #         same_diam[j] = True
  #   if np.any(same_diam):
  #     j = np.min(np.flatnonzero(same_diam))
  #     cofacet = np.sort(np.array([u,v,j], dtype=int))
  #     if np.all(cofacet == T):
  #       pair = (max_facet, comb_to_rank(cofacet, k=3, n=n, order="lex"))
  #       result.append(pair)
  # ap = np.array(result)
  # return(ap)

## From ripser paper: 
# pair_totals = np.array([18145, 32167, 230051, 2192209, 1386646, 122324, 893])
# non_ap = np.array([ 53, 576, 2466, 14006, 576, 438, 39 ])
# (pair_totals-non_ap)/pair_totals
# array([0.99707909, 0.98209345, 0.98928064, 0.99361101, 0.99958461, 0.99641935, 0.95632699])




class SpectralRI(LinearOperator):
  """Configurable linear operator useful for assessing the rank of 'lower-left' submatrices of the p-th simplicial boundary operator.

  An instance of this class stores two boundary matrices: one internal one representing the operator over the 'global' complex, and 
  one 'fitted' operator representing a (re)-weighted subset of the global one. 
  
  Both the actual local and global instances are private, stored in the 'D_' and '_D', respectively. To use the matrix, use the 'D' member. 
  
  The row/column indices of the 'global' should be considered immutable.

  If information is known about the pivot status of certain simplices, its status can be set as follows:
     0 <=> pivot status is unknown.
    +1 <=> simplex is positive (i.e. creator): it appears as the first entry in a persistence pair
    -1 <=> simplex is negative (i.e. destroyer): it appears as the second entry in a persistence pairs

  Positive p-simplices lie in the kernel of the p-th boundary operator and are paired with negative (p+1) simplices, 
  which correspondingly lie in image of the (p+1) boundary operator. Because of this, for the purpose of rank computation, 
  the p-chains of positive p-simplices can be pruned from the 'global' operator, as they only contribute to the nullity. 
  Similarly, lower-left sub-matrices of the operator having rows (columns) spanning *only* negative p (p+1, resp.) 
  simplices must be full rank, which can detected to speed up the computation. 

  Parameters: 
    S := complex-like. Can be made optional in the future
    p := Homology dimension of interest (required).
  
  Fields: 
    D := p-dimensional boundary operator, stored as a sparse COO array. Row/column indices are immutable. 
    p_weights := non-negative weights for the p-simplices.    
    q_weights := non-negative weights for the (p+1)-simplices.    
    shape := tuple with integer (# p, # p). 
    dtype := configurable dtype. Defaults to float32. 
  
  Members: 
    matvec(x) := 
  """
  def cns(self, C) -> int:
    return np.array(comb_to_rank(C, n=self.n, order='colex'), dtype=np.int64)

  def __init__(self, S, p: int):
    from copy import deepcopy
    P = range(p + 2)
    self.n = sx.card(S, 0)
    self.p = p
    self._weights = { q : np.ones(sx.card(S, q)) for q in P } 
    self._simplices = { q : self.cns(sx.faces(S, q)) if sx.card(S,q) > 0 else [] for q in P }
    self._status = { q : np.zeros(sx.card(S, q)) for q in P } 
    self._D = { q : sx.boundary_matrix(S, p=q).tocoo() for q in P }
    self.weights = deepcopy(self._weights)
    self.simplices = deepcopy(self._simplices)
    self.status = deepcopy(self._status)
    self.D = deepcopy(self._D)
    # list(run_length.encode(L._D[2].data))

    # self.weights = self._weights 
    # self.weights =
    # self.cns = { }
    # self.pr = np.array(comb_to_rank(sx.faces(S, p), n=self.n, order='colex'), dtype=np.int64)
    # self.qr = np.array(comb_to_rank(sx.faces(S, p+1), n=self.n, order='colex'), dtype=np.int64)
    # self._D = sx.boundary_matrix(S, p=p+1).tocoo() # NOTE: should never call eliminate_zeros() directly!
    # self.p_simplices = comb_to_rank(sx.faces(S, p), k=p+1, order='colex', n=sx.card(S,0))
    # self.q_simplices = comb_to_rank(sx.faces(S, p+1), k=p+2, order='colex', n=sx.card(S,0))
    # self._p_weights = np.ones(sx.card(S, p))
    # self._q_weights = np.ones(sx.card(S, p+1))
    # self._p_status = np.zeros(sx.card(S, p), dtype=np.int64)
    # self._q_status = np.zeros(sx.card(S, p+1), dtype=np.int64)
    self.shape = (sx.card(S,p), sx.card(S,p))
    self.dtype = np.dtype('float32')

  def _matvec(self, x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1)
    x *= self.weights[self.p]
    x = self.D[self.p] @ (self.weights[self.p+1] * (self.D[self.p].T @ x))
    x *= self.weights[self.p]
    return x

  def reset(self, weights: bool = False):
    for q in self._weights.keys():
      if weights:
        self._weights[q].fill(1)
      self._status[q].fill(0)
      N = len(self._simplices[q])
      self._D[q].data = np.repeat([(-1)**q for q in range(q+1)], N)
    self.shape = (len(self.weights[self.p]), len(self.weights[self.p]))
    # delattr(self, "D_")
    # delattr(self, "p_weights_")
    # delattr(self, "q_weights_")

  # def compressed_D(self, p: int, apparent: bool = False) -> sparray:
  #   """Removes (co)boundary (p/p+1)-chains lying in the nullspace of the operator, returning the compressed sparse matrix as a result.""" 
  #   r, c = self._D[p].row, self._D[p].col
  #   if len(self._weights[p]) > 0:
  #     self._D[p].data[self._weights[p][r] == 0] = 0
  #   if len(self._weights[p+1]) > 0:
  #     self._D[p].data[self._weights[p+1][c] == 0] = 0
  #   return deflate_sparse(self._D[p])

  def lower_left(self, i: float, j: float, p: int, deflate: bool = False, apparent: bool = False):
    """Modifies self to represent the p-th up Laplacian of the (p+1)-th boundary submatrix D_{p+1}[(i+1):,:j]  """
    ## Set the weights of the fixed complex to reflect (i,j)
    q = p + 1
    # self.reset(weights=False)
    # self._weights[p] = np.where(self._weights[p] > i, self._weights[p], 0)
    # self._weights[q] = np.where(self._weights[q] <= j, self._weights[q], 0) # np.logical_and(self._weights[q] > i, 
    
    ## This seems safe from a rank perspective
    # self._weights[p] = np.where(np.logical_and(self._weights[p] > i, self._weights[p] <= j), self._weights[p], 0)
    # self._weights[q] = np.where(np.logical_and(self._weights[q] > i, self._weights[q] <= j), self._weights[q], 0)
    ri, ci = self._D[q].row, self._D[q].col
    p_inc = np.logical_and(self._weights[p] > i, self._weights[p] <= j)
    q_inc = np.logical_and(self._weights[q] > i, self._weights[q] <= j)
    inc_mask = np.logical_and(p_inc[ri], q_inc[ci])
    
    # ri, ci = self._D[q].row, self._D[q].col
    # nullspace = np.logical_and(self._weights[q][ci] == 0, self._weights[p][ri] == 0)
    # self._D[q].data = np.where(nullspace, 0, self._D[q].data)
    # self._D[q].data[self._weights[q][ci] == 0] = 0
    # self._D[q].data[self._weights[p][ri] == 0] = 0


    # print(max(ci))
    # print(len(self._weights[q]))
    

    ## Zero elements outside of D[(i+1):,:j] based on the updated weights
    # ri, ci = self._D[q].row, self._D[q].col
    # self._D[q].data[self._weights[p][ri] == 0] = 0
    # self._D[q].data[self._weights[q][ci] == 0] = 0

    ## Update the cached weights + boundary matrices
    # self.weights[p] = np.extract(self._weights[p] > 0, self._weights[p])
    # self.weights[q] = np.extract(self._weights[q] > 0, self._weights[q])
    self.weights[p] = np.extract(p_inc, self._weights[p])
    self.weights[q] = np.extract(q_inc, self._weights[q])
    self.D[q] = deflate_sparse(self._D[q], inc_mask)
    # assert len(self.weights[p]) == self.D[q].shape[0], f"Incorrect weight lengths for the {p}-rows!"
    # assert len(self.weights[q]) == self.D[q].shape[1], f"Incorrect weight lengths for the {q}-cols!"

    ## Update the final shape
    self.shape = self.D[q].shape

    # self._weights[p][self._weights[p] <= i] = 0
    # self._weights[q][self._weights[q] > j] = 0
    # self.D[p] = self.compressed_D(p)

    # ## Populate the fitted attributes for the properties
    # ri = np.unique(self._D[p].row[self._D[p].data != 0])
    # ci = np.unique(self._D[p].col[self._D[p].data != 0])
    # self.weights[p] = np.take(self._weights[p], ri) if len(ri) > 0 else self._weights[p]
    # self.weights[q] = np.take(self._weights[q], ci) if len(ci) > 0 else self._weights[q] # self._q_weights > 0
    # assert len(self.weights[p]) == self.D[p].shape[0], "Failed to compress! This shouldn't happen."

    # self.shape = (self.D[p].shape[0], self.D[p].shape[0])
    return self 

  def rank(self, i: float, j: float, p: int):
    """Computes the numerical rank of a 'lower-left' sub-matrix of the p-th boundary operator, as determined by (p/q) weights."""
    q = p + 1
    p_inc = np.logical_and(self._weights[p] > i, self._weights[p] <= j)
    q_inc = np.logical_and(self._weights[q] > i, self._weights[q] <= j)
    
    ## First, check to see if the sub-matrix of interest consists solely of pivot entries 
    # self._weights[p] > i
    is_pivot_rows = self._status[p][p_inc] # positive p-simplices 
    is_pivot_cols = self._status[q][q_inc] # positive q-simplices
    if np.all(is_pivot_rows) or np.all(is_pivot_cols):
      print("shortcut taken")
      return min(len(is_pivot_rows), len(is_pivot_cols))

    ## If that fails, do a numerical rank computation 
    from primate.functional import numrank
    res = numrank(self.lower_left(i,j,p))
    return res

  def detect_pivots(self, X: np.ndarray, p: int, f_type: str = "lower"):
    """Searches for apparent pairs in the complex, flagging any found as 'pivots'"""
    from spirit.apparent_pairs import clique_mod
    assert f_type == "lower" or f_type == "flag"
    CT = clique_mod.Cliqueser_flag if f_type == "flag" else clique_mod.Cliqueser_lower
    C = CT(self.n, p+1)
    # q_ap = np.array([C.apparent_zero_facet(r, self.p+1) for r in self.qr])
    C.init(X) # format of X depends on f_type 
    p_ap = np.array([C.apparent_zero_cofacet(r, p) for r in self.simplices[p]])
    self._status[p][p_ap != -1] = p_ap[p_ap != -1] # set positive / save ranks of cofacets
    # self._q_status = q_ap

  def query(self, i: float, j: float, k: float = None, l: float = None) -> float:
    pass

  # @property
  # def D(self):
  #   ## If D has fitted operator, return that, otherwise return the original
  #   return self.D_ if hasattr(self, "D_") else self._D
  
  # @D.setter
  # def D(self, value):
  #   raise ValueError("Cannot change the boundary matrix once constructed")

  # @property
  # def p_weights(self):
  #   ## If has fitted attribute, return that, otherwise return the original
  #   return self.p_weights_ if hasattr(self, "p_weights_") else self._p_weights

  # @p_weights.setter
  # def p_weights(self, value):
  #   assert len(value) == len(self.p_weights) 
  #   self._p_weights = np.array(value).astype(self.dtype)

  # @property
  # def q_weights(self):
  #   ## If has fitted attribute, return that, otherwise return the original
  #   return self.q_weights_ if hasattr(self, "q_weights_") else self._q_weights

  # @q_weights.setter
  # def q_weights(self, value):
  #   assert len(value) == len(self.q_weights) 
  #   self._q_weights = np.array(value).astype(self.dtype)    
