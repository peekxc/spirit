import numpy as np
import splex as sx
import _clique as clique_mod

from typing import Callable, Optional, Union
from scipy.sparse import coo_array, sparray, coo_matrix, issparse, spmatrix
from scipy.sparse.linalg import LinearOperator

from scipy.special import comb
from itertools import combinations
from combin import rank_to_comb, comb_to_rank

from typing import * 
from .query import points_in_box, bisection_tree

## from: https://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r
def index_of(a: List[Hashable], b: List[Hashable], default: Any = None) -> List[int]:
  """Finds the index of each element A[i] in B """
  b_dict = {x: i for i, x in enumerate(b)}
  return np.array([b_dict.get(x, default) for x in a])

def deflate_sparse(A: sparray, mask: np.ndarray = None, ind: bool = False, sort_ind: bool = False):
  """'Deflates' a given sparse array 'A' by removing it's rows and columns that are all zero.   
  Returns a new sparse matrix with the same data but potentially diffferent shape. 
  """
  from hirola import HashTable
  A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
  hr = HashTable(1.25*A.shape[0] + 16, A.row.dtype)
  hc = HashTable(1.25*A.shape[1] + 16, A.col.dtype)
  non_zero = A.data != 0 if mask is None else mask
  assert len(non_zero) == len(A.data), "Mask invalid! Must be length of data array."
  r_nz = A.row[non_zero]
  c_nz = A.col[non_zero]
  d_nz = A.data[non_zero]
  if sort_ind:
    ri, ci = np.unique(r_nz), np.unique(c_nz)
    hr.add(ri)
    hc.add(ci)
    A_deflated = coo_matrix((d_nz, (hr[r_nz], hc[c_nz])), shape=(hr.length, hc.length))  ## Use matrix for sksparse
    A_deflated.eliminate_zeros()
    return A_deflated if not(ind) else (A_deflated, ri, ci)
  else: 
    hri = hr.add(r_nz) 
    hci = hc.add(c_nz) 
    A_deflated = coo_matrix((d_nz, (hri, hci)), shape=(hr.length, hc.length))
    A_deflated.eliminate_zeros()
    return A_deflated if not(ind) else (A_deflated, hr.keys, hc.keys)
  # return A_deflated if not(ind) else (A_deflated, ri, ci)

def compress_index(A: sparray, row_mask: np.ndarray, col_mask: np.ndarray, ind: bool = False):
  """Compresses a given sparse coo-matrix 'A' by keeping only the supplied row/column indices"""
  from hirola import HashTable
  from scipy.sparse import coo_matrix
  A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
  DP, (RI,CI) = clique_mod.compress_coo(row_mask, col_mask, A.row, A.col, A.data)
  hr = HashTable(1.25*len(RI) + 16, RI.dtype)
  hc = HashTable(1.25*len(CI) + 16, CI.dtype)
  hri = hr.add(RI)
  hci = hc.add(CI)
  A_deflated = coo_matrix((DP, (hri, hci)), shape=(hr.length, hc.length))
  A_deflated.eliminate_zeros()
  return A_deflated if not(ind) else (A_deflated, hr.keys, hc.keys)

class UpLaplacian(LinearOperator):
  def __init__(self, D: sparray, wp: np.ndarray, wq: np.ndarray):
    assert D.shape[0] == len(wp) and D.shape[1] == len(wq), "Dimension mismatch"
    self.D = D
    self.wp = wp
    self.wq = wq
    self.dtype = np.dtype("float32")
    self.shape = (D.shape[0], D.shape[0])

  def _matvec(self, x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1)
    x *= self.wp
    x = self.D @ (self.wq * (self.D.T @ x))
    x *= self.wp
    return x

  def tosparse(self):
    # from scipy.sparse import dia_array
    from scipy.sparse import dia_matrix
    n, m = len(self.wp), len(self.wq)
    WP = dia_matrix((np.array([self.wp]), [0]), shape=(n,n))
    WQ = dia_matrix((np.array([self.wq]), [0]), shape=(m,m))
    return WP @ self.D @ WQ @ self.D.T @ WP
  
  # TODO: to do this efficiently, really need a fast index remapper {0, 4, 5, 6, 6, 7} -> {0, 1, 2, 3, 3, 4}
  # def lower_left(self, a: float, b: float):
  #   """Partitions underlying boundary operator to reflect D[a:, :(b+1)]"""
  #   np.argpartition()
  #   self.wp >= a

def boundary_matrix(p: int, p_simplices: np.ndarray, f_simplices: np.ndarray = [], dtype=np.int8, n: int = None):
  """
  p = dimension of the p-simplices
  p_simplices = colex ranks of the p-simplices
  f_simplices = colex ranks of the (p-1)-simplices
  """
  if p <= 0: 
    return np.empty(shape=(0, len(p_simplices)), dtype=dtype)
  card_p, card_f = len(p_simplices), len(f_simplices)
  if card_f == 0: 
    raise ValueError("Not supported yet")
  if card_p == 0 or card_f == 0: 
    return np.empty(shape=(card_f, card_p), dtype=dtype)
  n = np.max(rank_to_comb(np.max(p_simplices), order='colex', k=p+1)) + 1 if n is None else n
  d, (ri,ci) = clique_mod.build_coo(n, p, p_simplices, f_simplices)
  D = coo_matrix((d, (ri,ci)), shape=(card_f, card_p), dtype=dtype)
  return D

class SpectralRI:
  """Spectral-approximation of the persistent rank invariant. 

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

  Based on the configured p, the corresponding matvec represents the action of the (p-1) up-Laplacian, which itself represents
  the Gram matrix of the weighted p co-chains (rows of D[p+1])

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

  def __init__(self, n: int, max_dim: int):
    self.n = n
    self.max_dim = max_dim
    P = range(self.max_dim + 1)
    self._weights = { q : [] for q in P } 
    self._simplices = { q : [] for q in P }
    self._status = { q : [] for q in P } 
    self._D = { q : [] for q in P } 

  def __repr__(self) -> str:
    max_dim = max(self._simplices.keys())
    msg = f"Spectral Rank Invariant up to {max_dim}-d\n"
    ns = tuple([len(s) for s in self._simplices.values()])
    nd = tuple(range(max_dim+1))
    msg += f"with {ns}-simplices of dimension {nd}"
    return msg

  def construct(self, X: np.ndarray, p: int = None, threshold: float = np.inf, apparent: bool = False, discard: bool = False, filter: str = "flag", **kwargs):
    """Constructs the simplices, weights, and pivot status of given filtration type up to *threshold*.
    
    Parameters: 
      X = point cloud, pairwise distances, or generic input type needed by 'filter'
      p = the dimension to construct. If not supplied, constructs all simplices up to 'max_dim'.
      threshold = filtration index to construct up to. 
      apparent = whether to 
      discard = 
    """
    from numbers import Number
    assert filter == "star" or filter == "flag" or filter == "metric"
    CM = clique_mod.__dict__['Cliqueser_' + filter]
    self.f_type = filter
    self.cm = CM(self.n, self.max_dim+1)
    self.cm.init(X) # format of X depends on f_type 
    # const size_t p, const float threshold, const bool check_pos = false, const bool check_neg = false, const bool filter_pos = false){
    P = range(self.max_dim + 1) if p is None else [int(p)]
    lb, ub = (-np.inf, threshold) if isinstance(threshold, Number) else threshold
    for p in P:
      p_simplices, p_weights, p_status = self.cm.build(p, lb, ub, apparent, apparent, discard)
      self._simplices[p] = p_simplices
      self._weights[p] = p_weights
      self._status[p] = p_status
    return self

  def boundary_matrix(self, p: int, dtype = np.float32):
    if p > self.max_dim: 
      raise ValueError(f"Invalid dimension p = '{p}' supplied.")
    if p <= 0: 
      return np.empty(shape=(0, len(self._simplices[0])), dtype=dtype)
    else:
      from scipy.sparse import coo_matrix
      card_p, card_f = len(self._simplices[p]), len(self._simplices[p-1])
      if card_p == 0 or card_f == 0: 
        return np.empty(shape=(card_f, card_p), dtype=dtype)
    d, (ri,ci) = clique_mod.build_coo(len(self._simplices[0]), p, self._simplices[p], self._simplices[p-1])
    D = coo_matrix((d, (ri,ci)), shape=(card_f, card_p), dtype=dtype)
    return D

  def reset(self, weights: bool = False):
    """Reset's the boundary matrix data, the weights, and the pivot status to their default initialized values."""
    for q in self._weights.keys():
      if weights:
        self._weights[q].fill(1)
      self._status[q].fill(0)
      N = len(self._simplices[q])
      self._D[q].data = np.repeat([(-1)**q for q in range(q+1)], N)

  def lower_left(self, i: float, j: float, p: int, deflate: bool = False, apparent: bool = False, expand: bool = False):
    """Modifies both the (p / p - 1)-weights and D[p] to represent the lower left submatrix D_{p}[(i+1):,:j]."""
    assert issparse(self._D[p]), "p-th boundary matrix not found. Has it been constructed?"

    ## This seems safe from a rank perspective
    f = p - 1
    f_inc = np.logical_and(self._weights[f] >= i, self._weights[f] <= j)
    p_inc = np.logical_and(self._weights[p] >= i, self._weights[p] <= j)
    
    ## If requested, also check status for apparent pairs, removing them when known
    if apparent:
      # f_inc[self._status[f] > 0] = False
      p_inc[self._status[p] > 0] = False

    ## Update the cached weights + boundary matrices
    ## See: https://stackoverflow.com/questions/71225872/why-does-numpy-viewbool-makes-numpy-logical-and-significantly-faster
    if deflate:     
      if expand: 
        ri, ci = self._D[p].row, self._D[p].col
        inc_mask = f_inc[ri].view(bool) & p_inc[ci].view(bool) # explicit index expansion 
        Dp, ri_inc, ci_inc = deflate_sparse(self._D[p], inc_mask, ind=True)
      else:
        Dp, ri_inc, ci_inc = compress_index(self._D[p], f_inc, p_inc, ind=True)
      wf = self._weights[f][ri_inc]
      wp = self._weights[p][ci_inc]
    else: 
      ri, ci = self._D[p].row, self._D[p].col
      inc_mask = f_inc[ri].view(bool) & p_inc[ci].view(bool)
      Dp = self._D[p].copy()
      Dp.data = np.where(inc_mask, Dp.data, 0.0)
      wf = np.where(f_inc, self._weights[f], 0.0)
      wp = np.where(p_inc, self._weights[p], 0.0)
    assert len(wf) == Dp.shape[0], f"Incorrect weight lengths ({len(wf)}) for # of {f}-rows! ({Dp.shape[0]})"
    assert len(wp) == Dp.shape[1], f"Incorrect weight lengths ({len(wp)}) for # of {p}-cols! ({Dp.shape[1]})"
    return UpLaplacian(Dp, wf, wp)

  def rank(self, p: int, a: float, b: float, method: str = ["direct", "cholesky", "trace"], **kwargs):
    """Computes the numerical rank of the (>= a, <= b)-lower-left submatrix of the p-th boundary operator."""
    if p <= 0: 
      return 0
    f = p - 1
    f_inc = np.logical_and(self._weights[f] >= a, self._weights[f] <= b)
    p_inc = np.logical_and(self._weights[p] >= a, self._weights[p] <= b)
    
    ## Degenerate case
    if np.sum(f_inc) == 0 or np.sum(p_inc) == 0:
      return 0 

    ## First, check to see if the sub-matrix of interest consists solely of pivot entries 
    is_pivot_rows = self._status[f][f_inc] < 0 # negative p-simplices 
    is_pivot_cols = self._status[p][p_inc] < 0 # negative q-simplices
    if np.all(is_pivot_rows) or np.all(is_pivot_cols):
      print("apparent full rank shortcut taken")
      return min(len(is_pivot_rows), len(is_pivot_cols))

    ## Start with a matrix-free Up Laplacian operator 
    LA = self.lower_left(a, b, p, deflate=True, apparent=True)
    if np.prod(LA.shape) == 0:
      return 0 

    ## Try to first detect full rank via logdet 
    # from primate.trace import hutch
    # hutch(LA, deg=LA.shape[0], orth=, maxiter=5)
    if method == "direct" or method == ["direct", "cholesky", "trace"]:
      return np.linalg.matrix_rank(LA.D.todense(), **kwargs)
    elif method == "cholesky": 
      from sksparse.cholmod import cholesky_AAt
      kwargs['beta'] = kwargs.get('beta', 1e-6)
      Dp_csc = LA.D.tocsc()
      F = cholesky_AAt(Dp_csc, **kwargs).D()
      threshold = max(np.max(F) * max(LA.D.shape) *  np.finfo(np.float32).eps, kwargs['beta'] * 100)
      return np.sum(F > threshold)
    elif method == "trace":
      from primate.functional import numrank
      return numrank(LA, **kwargs)
    else:
      raise ValueError(f"Invalid method '{method}' supplied; must be one of 'direct', 'cholesky', or 'trace.'")
  
  def spectral_sum(self, p: int, a: float, b: float, fun: Union[str, Callable], method: str = ["trace", "direct"], **kwargs):
    """Computes the spectral sum of the (a,b)-lower-left submatrix of the p-th boundary operator."""
    from primate.functional import hutch
    LA = self.lower_left(a, b, p, deflate=True, apparent=True) 
    if np.prod(LA.shape) == 0:
      return 0 

    ## USe either direct calculation or stochastic trace call
    if method == "trace" or method == ["direct", "trace"]:
      assert method == "trace", "Invalid method specified"
      return hutch(LA, fun=fun, **kwargs)
    else: 
      assert isinstance(fun, Callable), "'fun' must be callable"
      ew = np.linalg.eigvalsh(LA.tosparse().todense())
      return np.sum(fun(ew))

  def query(self, p: int, a: float, b: float, c: float = None, d: float = None, delta: float = 0, summands: bool = False, **kwargs) -> float:
    """Queries the number of persistent pairs from Hp that intersect the box (a,b] x (c,d].
    
    Note that if multiple pairs lie exactly on the boundary of the box, only a fraction of them will be reported. 
    """
    q = p + 1
    if (c is None and d is None) or (c == -np.inf and d == np.inf):
      terms = [0]*4
      terms[0] = np.sum(self._weights[p] <= a)
      terms[1] = self.rank(p, -np.inf, a, **kwargs)
      terms[2] = self.rank(q, -np.inf, b, **kwargs)
      terms[3] = self.rank(q, a+delta, b, **kwargs)
      return sum(s*t for s,t in zip([+1,-1,-1,+1], terms)) if not(summands) else terms
      # raise NotImplementedError("not implemented yet")
    else:
      pairs = [(b+delta,c), (a+delta,c), (b+delta,d), (a+delta,d)] 
      # pattern = [(1,1),(1,0),(0,1),(0,0)]
      # terms = [self.rank(q, i+x*delta, j-y*delta, **kwargs) for cc, (x,y) in enumerate(pattern)]
      terms = [self.rank(q, i, j, **kwargs) for i,j in pairs]
      return sum(s*t for s,t in zip([+1,-1,-1,+1], terms)) if not(summands) else terms

  def query_spectral(self, p: int, a: float, b: float, c: float = None, d: float = None, summands: bool = False, fun: Callable = np.sign, **kwargs):
    """Queries the dimension of the persistent homology class H_p(a,b,c,d). """
    q = p + 1
    if (c is None and d is None) or (c == -np.inf and d == np.inf):
      terms = [0]*4
      terms[0] = np.sum(fun(self._weights[p][self._weights[p] <= a]))
      terms[1] = self.spectral_sum(p, 0, a, fun, **kwargs)
      terms[2] = self.spectral_sum(q, 0, b, fun, **kwargs)
      terms[3] = self.spectral_sum(q, a, b, fun, **kwargs)
      return sum(s*t for s,t in zip([+1,-1,-1,+1], terms)) if not(summands) else terms
      # raise NotImplementedError("not implemented yet")
    else:
      pairs = [(b,c), (a,c), (b,d), (a,d)] 
      terms = [self.spectral_sum(q, i, j, fun, **kwargs) for cc, (i,j) in enumerate(pairs)]
      return sum(s*t for s,t in zip([+1,-1,-1,+1], terms)) if not(summands) else terms

  def _index_weights(self):
    D = len(self._weights)
    weights = np.hstack([self._weights[q] for q in range(D)])
    ranks = np.hstack([self._simplices[q] for q in range(D)])
    dims = np.hstack([np.repeat(q, len(self._simplices[q])) for q in range(D)])
    wrd = np.array(list(zip(weights, ranks, dims)), dtype=[('weight', 'f4'), ('rank', 'i8'), ('dim', 'i4')])
    # wrd['rank'] = -wrd['rank']
    wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank'))) #  + 1
    index_weights = { q : wrd_ranking[dims == q] for q in range(D) }
    return index_weights, wrd

  ## Queries the persistent pairs via a logarithmic number of rank computations on the index persistence plane 
  ## Iteratively re-weights 
  def query_pairs(self, p: int, a: float, b: float, c: float, d: float, delta: float = 1e-15, simplex_pairs: bool = False, **kwargs):
    """Queries the persistent pairs via a logarithmic number of rank computations on the index persistence plane."""
    assert a < b and b <= c and c < d, f"Invalid box ({a},{b}]x({c},{d}] given; must satisfy a < b <= c < d!" # see eq. 2.1 in the persistent measure paper 
    self.query(p,a,b,c,d)
    
    weights_backup = self._weights.copy()
    self._weights, wrd = self._index_weights()
    verbose: bool = kwargs.pop("verbose", False)

    ## The real-valued weights, as a vector 
    rw = np.hstack([weights_backup[q] for q in range(len(self._weights))])
    iw = np.hstack([self._weights[q] for q in range(len(self._weights))])
    # np.argsort(np.argsort(rw))

    ## Translate the query into a *valid* query on the index persistence plane 
    ## NOTE: The offsets are needed because we cannot handle degenerate boxes on index-persistence
    ## We use sum instead of searchsorted because the weights are not ordered!
    bi = np.sum(b >= rw) # np.sum(b >= rw) - 1 ## the left is a the tightest inclusive query (rounding down), the right is one more
    di = np.sum(d >= rw)  # np.sum(d >= rw) -1 ## The right works on the index persistence plane
    ai = max(np.sum(a >= rw) - 1, 0)# max(min(np.sum(rw <= a) - 1, bi-1), 0)
    ci = max(np.sum(c >= rw) - 1, bi) # max(min(np.sum(rw <= c) - 1, di-1), bi)

    ## Checks
    rw_sorted = np.sort(rw)
    assert b <= rw_sorted[bi], f"Index mapped b={b} must snap right to {rw_sorted[bi]}"
    if bi > 0:
      assert rw_sorted[bi-1] <= b, f"Index mapped b={b} must be larger than {rw_sorted[bi-1]}"
    
    assert rw_sorted[ai] <= a, f"Index mapped a={a} must snap left to {rw_sorted[ai]}"
    if bi > 0:
      assert rw_sorted[bi-1] <= b, f"Index mapped b={b} must be larger than {rw_sorted[bi-1]}"
    # assert rw_sorted[di] <= d, f"Index mapped b={b} must snap left to {rw_sorted[bi]}"
    # # if di < (len(rw_sorted) - 1):
    # #   assert d < rw_sorted[di+1], f"Index mapped d={d} must be less than outer index {rw_sorted[di+1]}"
    # assert True if di == 0 else rw_sorted[di-1] < d and d <= rw_sorted[di], "Index mapped b must include the b interval"
    print(f"Index translated box: [{ai}, {bi}] x [{ci}, {di}]")
    print(f"Function translated box: [{rw_sorted[ai]:.3f}, {rw_sorted[bi]:.3f}] x [{rw_sorted[ci]:.3f}, {rw_sorted[di]:.3f}]")

    ## Do the divide and conquer pair search
    kwargs['method'] = kwargs.get('method', 'cholesky')
    pairs = points_in_box(ai,bi,ci,di,lambda i,j,k,l: self.query(p,i,j,k,l,**kwargs), verbose)
    if verbose:
      print(f"Index translated box: [{ai}, {bi}] x [{ci}, {di}]")
      print(pairs)
    
    ## Reformat the pairs in the function persistence plane
    q = p + 1
    p_dgm = []
    for pos_ind, neg_ind in pairs.items():
      pos_idx = np.flatnonzero(self._weights[p] == pos_ind)
      neg_idx = np.flatnonzero(self._weights[q] == neg_ind)
      assert len(pos_idx) == 1 and len(neg_idx) == 1, "Failed to re-index the pairs"
      pw, rp = weights_backup[p][pos_idx].item(), self._simplices[p][pos_idx]
      nw, rq = weights_backup[q][neg_idx].item(), self._simplices[q][neg_idx] 
      p_dgm.append([pw, nw] if not simplex_pairs else [pw, nw, rp, rq])
    self._weights = weights_backup
    dgm_dtype = [("birth", "f4"), ("death", "f4")] if not simplex_pairs else [("birth", "f4"), ("death", "f4"), ("creator", "i8"), ("destroyer", "i8")]
    return np.array([tuple(pair) for pair in p_dgm], dtype=dgm_dtype)


