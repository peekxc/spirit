from itertools import combinations
from numbers import Integral, Number
from typing import *
from typing import Any, Callable, Hashable, List, Optional, Union

import _clique as clique_mod
import numpy as np
from combin import comb_to_rank, rank_to_comb
from scipy.sparse import coo_array, coo_matrix, issparse, sparray, spmatrix
from scipy.sparse.linalg import LinearOperator
from scipy.special import comb

from .query import bisection_tree, points_in_box

DEBUG_VAR = None


## TODO
class LaplacianSparse:
	pass


## from: https://stackoverflow.com/questions/4110059/pythonor-numpy-equivalent-of-match-in-r
def index_of(a: List[Hashable], b: List[Hashable], default: Any = None) -> List[int]:
	"""Finds the index of each element A[i] in B"""
	b_dict = {x: i for i, x in enumerate(b)}
	return np.array([b_dict.get(x, default) for x in a])


## Converts a sparse matrix into canonical format
def to_canonical(A: sparray, form: str = "csr"):
	m_type = form.lower()
	method_call = "to" + m_type
	A = getattr(A, method_call)()
	if m_type in ["csr", "csc"]:
		A.sort_indices()  # ensure sorted indices
		A.eliminate_zeros()  # remove zeros
		A.sum_duplicates()  # eliminate duplicate entries
		A.prune()  # remove empty-space
		assert A.has_canonical_format, "Failed to convert to canonical format"
	elif m_type == "coo":
		A.eliminate_zeros()
		A.sum_duplicates()
	return A


def deflate_sparse(A: sparray, mask: np.ndarray = None, ind: bool = False, sort_ind: bool = False):
	"""'Deflates' a given sparse array 'A' by removing it's rows and columns that are all zero.
	Returns a new sparse matrix with the same data but potentially diffferent shape."""
	from hirola import HashTable

	A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
	hr = HashTable(1.25 * A.shape[0] + 16, A.row.dtype)
	hc = HashTable(1.25 * A.shape[1] + 16, A.col.dtype)
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
		return A_deflated if not (ind) else (A_deflated, ri, ci)
	else:
		hri = hr.add(r_nz)
		hci = hc.add(c_nz)
		A_deflated = coo_matrix((d_nz, (hri, hci)), shape=(hr.length, hc.length))
		A_deflated.eliminate_zeros()
		return A_deflated if not (ind) else (A_deflated, hr.keys, hc.keys)
	# return A_deflated if not(ind) else (A_deflated, ri, ci)


def compress_index(A: sparray, row_mask: np.ndarray, col_mask: np.ndarray, ind: bool = False):
	"""Compresses a given sparse coo-matrix 'A' by keeping only the supplied row/column indices"""
	from hirola import HashTable
	from scipy.sparse import coo_matrix

	A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
	DP, (RI, CI) = clique_mod.compress_coo(row_mask, col_mask, A.row, A.col, A.data)
	hr = HashTable(1.25 * len(RI) + 16, RI.dtype)
	hc = HashTable(1.25 * len(CI) + 16, CI.dtype)
	hri = hr.add(RI)
	hci = hc.add(CI)
	A_deflated = coo_matrix((DP, (hri, hci)), shape=(hr.length, hc.length))
	A_deflated.eliminate_zeros()
	return A_deflated if not (ind) else (A_deflated, hr.keys, hc.keys)

	# TODO: to do this efficiently, really need a fast index remapper {0, 4, 5, 6, 6, 7} -> {0, 1, 2, 3, 3, 4}
	# def lower_left(self, a: float, b: float):
	#   """Partitions underlying boundary operator to reflect D[a:, :(b+1)]"""
	#   np.argpartition()
	#   self.wp >= a


def boundary_matrix(p: int, p_simplices: np.ndarray, f_simplices: np.ndarray = [], dtype=np.int16, n: int = None):
	"""
	p = dimension of the p-simplices
	p_simplices = colex ranks of the p-simplices
	f_simplices = colex ranks of the (p-1)-simplices
	"""
	if p <= 0:
		return coo_matrix((0, len(p_simplices)), dtype=dtype)
	card_p, card_f = len(p_simplices), len(f_simplices)
	if card_p == 0 or card_f == 0:
		return coo_matrix((card_f, card_p), dtype=dtype)
	n = np.max(rank_to_comb(np.max(p_simplices), order="colex", k=p + 1)) + 1 if n is None else n
	d, (ri, ci) = clique_mod.build_coo(n, p, p_simplices, f_simplices)
	D = coo_matrix((d, (ri, ci)), shape=(card_f, card_p), dtype=dtype)
	return D


class UpLaplacian(LinearOperator):
	# def __init__(self, p: int, S: np.ndarray, F: np.ndarray, ws: np.ndarray = None, wf: np.ndarray = None):
	#   self.D = boundary_matrix(p, S, F)
	#   self.wp = np.ones(len(F)) if wf is None else np.array(wf)
	#   self.wq = np.ones(len(S)) if ws is None else np.array(ws)
	#   self.dtype = np.dtype("float64")
	#   self.shape = (self.D.shape[0], self.D.shape[0])

	def __init__(self, D: sparray, wp: np.ndarray = None, wq: np.ndarray = None):
		self.wp = np.ones(D.shape[0]) if wp is None else np.array(wp)
		self.wq = np.ones(D.shape[1]) if wq is None else np.array(wq)
		assert D.shape[0] == len(self.wp) and D.shape[1] == len(self.wq), "Dimension mismatch"
		self.D = to_canonical(D, "csr")
		self.Dt = to_canonical(D.T, "csr")
		self.dtype = np.dtype("float64")
		self.shape = (D.shape[0], D.shape[0])

	def _matvec(self, x: np.ndarray) -> np.ndarray:
		x = x.reshape(-1)
		x *= self.wp
		x = self.D @ (self.wq * (self.Dt @ x))
		x *= self.wp
		return x

	# def __repr__(self) -> str:
	#   return "UpLaplacian over ({}/{})  ({}/{})-simplices"

	def tosparse(self):
		# from scipy.sparse import dia_array
		from scipy.sparse import dia_matrix

		n, m = len(self.wp), len(self.wq)
		WP = dia_matrix((np.array([self.wp]), [0]), shape=(n, n))
		WQ = dia_matrix((np.array([self.wq]), [0]), shape=(m, m))
		return WP @ self.D @ WQ @ self.Dt @ WP

	def diagonal(self):
		return np.diff(self.D.indptr)  ## works with CSR only


def subset_boundary(
	D: sparray,
	p_indices: np.ndarray,
	q_indices: np.ndarray,
	deflate: bool = True,
	expand: bool = False,
	wp: np.ndarray = None,
	wq: np.ndarray = None,
):
	"""Produces a subset of sparse boundary matrix 'D'

	Parameters:
	  p_indices = boolean vector indicating which rows to include
	  q_indices = boolean vector indicating which cols to include

	"""
	D = D.tocoo() if not hasattr(D, "row") and hasattr(D, "col") else D
	p_indices, q_indices = np.asarray(p_indices), np.asarray(q_indices)
	assert len(p_indices) == D.shape[0] and p_indices.dtype == np.dtype("bool")
	assert len(q_indices) == D.shape[1] and q_indices.dtype == np.dtype("bool")

	## Extract weights
	wp = np.ones(D.shape[0]) if wp is None else np.asarray(wp)
	wq = np.ones(D.shape[1]) if wq is None else np.asarray(wq)

	## Update the cached weights + boundary matrices
	## See: https://stackoverflow.com/questions/71225872/why-does-numpy-viewbool-makes-numpy-logical-and-significantly-faster
	if deflate:
		if expand:
			ri, ci = D.row, D.col
			inc_mask = p_indices[ri].view(bool) & q_indices[ci].view(bool)  # explicit index expansion
			Dp, ri_inc, ci_inc = deflate_sparse(D, inc_mask, ind=True)
		else:
			Dp, ri_inc, ci_inc = compress_index(D, p_indices, q_indices, ind=True)
		wf = wp[ri_inc]
		wp = wq[ci_inc]
	else:
		inc_mask = p_indices[D.row].view(bool) & q_indices[D.col].view(bool)  # explicit index expansion
		Dp = D.copy()
		Dp.data = np.where(inc_mask, Dp.data, 0.0)
		wf = np.where(p_indices, wp, 0.0)
		wp = np.where(q_indices, wq, 0.0)
	assert len(wf) == Dp.shape[0], f"Incorrect weight lengths ({len(wf)}) for # of rows! ({Dp.shape[0]})"
	assert len(wp) == Dp.shape[1], f"Incorrect weight lengths ({len(wp)}) for # of cols! ({Dp.shape[1]})"
	return (Dp, wf, wp)


# class MatrixFreeLaplacian(LinearOperator):
#   from comb_laplacian import LaplacianSparse
#   def __init__(self,
#     S: np.ndarray, F: np.ndarray,
#     n: int, k: int,
#     precompute_deg: bool = True,
#     gpu: bool = False, threadsperblock: int = 32, n_kernels: int = 1,
#     wp: np.ndarray = None, wq: np.ndarray = None
#   ):
#     self.op = LaplacianSparse(S, F, n, k, precompute_deg, gpu, threadsperblock, n_kernels)
#     self.wp = np.ones(len(F)) if wp is None else np.array(wp)
#     self.wq = np.ones(len(S)) if wq is None else np.array(wq)

#   def _matmat(self, X: np.ndarray) -> np.ndarray:
#     return self.op @ X

#   def lower_left(self, i: float, j: float):
#     self.op

# from comb_laplacian import LaplacianSparse
# ## This seems safe from a rank perspective
# ## THIS IS WRONG: should make generic lower_left not a method
# f_inc = np.logical_and(p_indices >= i, p_indices <= j)
# p_inc = np.logical_and(q_indices >= i, q_indices <= j)


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
	  n = number of vertices in the complex.
	  max_dim = maximum homology dimension of interest (required).
	  dtype = appropriate dtype for the coefficient field of choice (defaults to np.float64)

	Fields:
	  op := p-dimensional boundary operator, stored as a sparse COO array. Row/column indices are immutable.
	  weights = non-negative weights for the simplices up to dimension max_dim. See details.
	  simplices = the
	"""

	def cns(self, C) -> int:
		return np.array(comb_to_rank(C, n=self.n, order="colex"), dtype=np.int64)

	def __init__(self, n: int, max_dim: int, dtype=np.float64):
		assert isinstance(n, Integral), f"Number of vertices 'n' must be integer (supplied {n})"
		assert isinstance(max_dim, Integral), f"Number of vertices 'n' must be integer (supplied {max_dim})"
		self.n = n
		self.max_dim = max_dim
		P = range(self.max_dim + 1)
		self._weights = {q: np.empty_like([]) for q in P}
		self._simplices = {q: np.empty_like([]) for q in P}
		self._status = {q: np.empty_like([]) for q in P}
		# self._D = { q : [] for q in P }
		self._ops = {q: np.empty_like([]) for q in P}  # the laplacians
		self._dtype = np.finfo(dtype).dtype

	def __repr__(self) -> str:
		max_dim = max(self._simplices.keys())
		msg = f"Spectral Rank Invariant up to {max_dim}-d\n"
		ns = tuple([len(s) for s in self._simplices.values()])
		nd = tuple(range(max_dim + 1))
		msg += f"with {ns}-simplices of dimension {nd}"
		return msg

	def construct_flag(
		self,
		X: np.ndarray,
		max_dim: int = 1,
		threshold: float = np.inf,
		apparent: bool = True,
		discard_pos: bool = True,
		**kwargs,
	):
		"""Constructs the simplices, weights, and pivot status of given filtration type up to *threshold*.

		Parameters:
		  X = point cloud, pairwise distances, or generic input type needed by 'filter'
		  p = the dimension to construct. If not supplied, constructs all simplices up to 'max_dim'.
		  threshold = filtration index to construct up to.
		  apparent = whether to identify apparent pairs in each dimension
		  discard_pos = whether to discard positive simplices in the last dimension
		"""
		from numbers import Number

		from scipy.spatial.distance import pdist

		# assert filter == "star" or filter == "flag" or filter == "metric"
		self.f_type = "flag"
		CM = clique_mod.__dict__["Cliqueser_flag"]
		cm = CM(self.n, max_dim + 1)
		cm.init(pdist(X))  # format of X depends on f_type
		# const size_t p, const float threshold, const bool check_pos = false, const bool check_neg = false, const bool filter_pos = false){
		lb, ub = (-np.inf, threshold) if isinstance(threshold, Number) else threshold
		for p in range(max_dim + 2):
			discard = False if p != (max_dim + 1) else discard_pos
			p_simplices, p_weights, p_status = cm.build(p, lb, ub, apparent, apparent, discard)
			self._simplices[p] = p_simplices
			self._weights[p] = p_weights
			self._status[p] = p_status
		# for p in P:
		#   p_simplices, p_weights, p_status = self.cm.build(p, lb, ub, apparent, apparent, discard)
		#   self._simplices[p] = p_simplices
		#   self._weights[p] = p_weights
		#   self._status[p] = p_status
		return self

	def construct_operator(self, p: int, form: str = "boundary", **kwargs):
		"""Constructs the p-UpLaplacian over the (p/p+1)-simplices"""
		F = self._simplices[p].copy()
		S = np.asarray(self._simplices.get(p + 1, []).copy())
		if form == "matrix free":
			self._ops[p] = LaplacianSparse(S, F, n=self.n, k=p + 2, **kwargs)
		elif form == "boundary":
			D = boundary_matrix(p + 1, p_simplices=S, f_simplices=F)
			self._ops[p] = UpLaplacian(D)
		elif form == "sparse":
			raise NotImplementedError("Not implemented yet")
		else:
			raise ValueError("Invalid operator")

	def boundary_matrix(self, p: int, dtype=np.float64):
		if p > self.max_dim:
			raise ValueError(f"Invalid dimension p = '{p}' supplied.")
		if p <= 0:
			return np.empty(shape=(0, len(self._simplices[0])), dtype=dtype)
		else:
			card_p, card_f = len(self._simplices[p]), len(self._simplices[p - 1])
			if card_p == 0 or card_f == 0:
				return np.empty(shape=(card_f, card_p), dtype=dtype)
		d, (ri, ci) = clique_mod.build_coo(len(self._simplices[0]), p, self._simplices[p], self._simplices[p - 1])
		D = coo_matrix((d, (ri, ci)), shape=(card_f, card_p), dtype=dtype)
		return D

	# def reset(self, weights: bool = False):
	#   """Reset's the boundary matrix data, the weights, and the pivot status to their default initialized values."""
	#   for q in self._weights.keys():
	#     if weights:
	#       self._weights[q].fill(1)
	#     self._status[q].fill(0)
	#     N = len(self._simplices[q])
	#     self._D[q].data = np.repeat([(-1)**q for q in range(q+1)], N)

	def lower_left(self, p: int, i: float, j: float):
		"""Returns a Laplacian matrix from (p / p + 1)-simplices.

		The Laplacian has rank equal to the lower-left boundary submatrix D_p[(i+1):,:j]
		"""
		if isinstance(self._ops[p], UpLaplacian):
			# print(f"Getting {p}-Laplacian")
			q = p + 1
			p_inc = np.logical_and(self._weights[p] >= i, self._weights[p] <= j)
			q_inc = np.logical_and(self._weights[q] >= i, self._weights[q] <= j)
			L = UpLaplacian(*subset_boundary(self._ops[p].D, p_inc, q_inc))
			return L
		elif isinstance(self._ops[p], LaplacianSparse):
			q = p + 1
			p_inc = np.logical_and(self._weights[p] >= i, self._weights[p] <= j)
			q_inc = np.logical_and(self._weights[q] >= i, self._weights[q] <= j)
			F_ll = self._simplices[p][p_inc]
			S_ll = self._simplices[q][q_inc]
			print(F_ll)
			if len(S_ll) == 0 or len(F_ll) == 0:
				return np.empty(shape=(len(F_ll), F_ll))
			print(f"S length: {len(S_ll)}, F length: {len(F_ll)}")
			return LaplacianSparse(S_ll, F_ll, self.n, p + 2)
		else:
			raise ValueError(f"Invalid Laplacian operator '{type(self._ops[p])}' specified")

		# RI._ops[1].lower_left(a=0.6, b=0.8)

		# assert issparse(self._D[p]), "p-th boundary matrix not found. Has it been constructed?"

		# ## This seems safe from a rank perspective
		# f = p - 1
		# f_inc = np.logical_and(self._weights[f] >= i, self._weights[f] <= j)
		# p_inc = np.logical_and(self._weights[p] >= i, self._weights[p] <= j)

		# ## If requested, also check status for apparent pairs, removing them when known
		# if apparent:
		#   # f_inc[self._status[f] > 0] = False
		#   p_inc[self._status[p] > 0] = False

		# ## Update the cached weights + boundary matrices
		# ## See: https://stackoverflow.com/questions/71225872/why-does-numpy-viewbool-makes-numpy-logical-and-significantly-faster
		# if deflate:
		#   if expand:
		#     ri, ci = self._D[p].row, self._D[p].col
		#     inc_mask = f_inc[ri].view(bool) & p_inc[ci].view(bool) # explicit index expansion
		#     Dp, ri_inc, ci_inc = deflate_sparse(self._D[p], inc_mask, ind=True)
		#   else:
		#     Dp, ri_inc, ci_inc = compress_index(self._D[p], f_inc, p_inc, ind=True)
		#   wf = self._weights[f][ri_inc]
		#   wp = self._weights[p][ci_inc]
		# else:
		#   ri, ci = self._D[p].row, self._D[p].col
		#   inc_mask = f_inc[ri].view(bool) & p_inc[ci].view(bool)
		#   Dp = self._D[p].copy()
		#   Dp.data = np.where(inc_mask, Dp.data, 0.0)
		#   wf = np.where(f_inc, self._weights[f], 0.0)
		#   wp = np.where(p_inc, self._weights[p], 0.0)
		# assert len(wf) == Dp.shape[0], f"Incorrect weight lengths ({len(wf)}) for # of {f}-rows! ({Dp.shape[0]})"
		# assert len(wp) == Dp.shape[1], f"Incorrect weight lengths ({len(wp)}) for # of {p}-cols! ({Dp.shape[1]})"
		# return UpLaplacian(Dp, wf, wp)

	def rank(self, p: int, a: float, b: float, method: str = "cholesky", **kwargs):
		"""Computes the numerical rank of the (>= a, <= b)-lower-left submatrix of the p-th boundary operator"""
		if p <= 0:
			return 0
		f = p - 1
		f_inc = np.logical_and(self._weights[f] >= a, self._weights[f] <= b)
		p_inc = np.logical_and(self._weights[p] >= a, self._weights[p] <= b)

		## Degenerate case
		if np.sum(f_inc) == 0 or np.sum(p_inc) == 0:
			return 0

		## First, check to see if the sub-matrix of interest consists solely of pivot entries
		is_pivot_rows = self._status[f][f_inc] < 0  # negative p-simplices
		is_pivot_cols = self._status[p][p_inc] < 0  # negative q-simplices
		if np.all(is_pivot_rows) or np.all(is_pivot_cols):
			print(f"Apparent full rank shortcut taken over {len(f_inc)} row and {len(is_pivot_cols)} cols")
			return min(len(is_pivot_rows), len(is_pivot_cols))

		## Start with a matrix-free Up Laplacian operator
		# LA = self.lower_left(a, b, p, deflate=True, apparent=True)
		# if np.prod(LA.shape) == 0:
		#   return 0
		# LA = self._ops[p].lower_left(a,b)
		LA = self.lower_left(f, a, b)
		# print(LA)

		## TODO: Try to first detect full rank via logdet?
		# print(f"Method = {method}")
		if method == "direct" or method == ["direct", "cholesky", "trace"]:
			assert isinstance(LA, UpLaplacian), "Direct method only support for UpLaplacian"
			return np.linalg.matrix_rank(LA.D.todense(), **kwargs)
		elif method == "cholesky":
			assert isinstance(LA, UpLaplacian), "Cholesky only support for UpLaplacian"
			## https://discourse.julialang.org/t/tolerance-specification-for-pivoted-cholesky/6863/2
			from sksparse.cholmod import cholesky_AAt

			kwargs["beta"] = kwargs.get("beta", 1e-6)
			Dp_csc = LA.D.tocsc()
			if np.prod(Dp_csc.shape) == 0:
				return 0
			if 1 in Dp_csc.shape:
				return int(np.any(Dp_csc.data != 0))
			F = np.sort(cholesky_AAt(Dp_csc, **kwargs).D())
			F_diff = np.diff(F) / F[:-1]
			global DEBUG_VAR
			## TODO: Write a custom detect_gap function
			## if there exists a value near beta and there's a jump that is two orders in magnitude
			if max(F_diff) > 10.0:  # min(F) < (kwargs['beta'] * 100) and
				nullity = np.argmax(F_diff) + 1
				m_rank = len(F) - nullity
				# if m_rank != np.linalg.matrix_rank(LA.D.todense()):
				DEBUG_VAR = LA
				# print("here1")
				return m_rank
			else:
				DEBUG_VAR = LA
				# print("here2")
				threshold = np.max(F) * max(LA.D.shape) * np.finfo(np.float64).eps
				return np.sum(F > threshold)
		elif method == "trace":
			from primate.functional import numrank
			from primate.trace import hutch

			# print(kwargs)
			tr_est = hutch(LA) if not hasattr(LA, "diagonal") else np.sum(LA.diagonal())
			gap_est = max(0.5 * (2.0 / tr_est) ** 2, 2 / LA.shape[0] ** 4)
			return numrank(LA, gap=gap_est, **kwargs)
		else:
			raise ValueError(f"Invalid method '{method}' supplied; must be one of 'direct', 'cholesky', or 'trace.'")

	def spectral_sum(
		self, p: int, a: float, b: float, fun: Union[str, Callable], method: str = ["trace", "direct"], **kwargs
	):
		"""Computes the spectral sum of the (a,b)-lower-left submatrix of the p-th boundary operator."""

		LA = self.lower_left(p, a, b)
		if np.prod(LA.shape) == 0:
			return 0

		## USe either direct calculation or stochastic trace call
		if method == "trace" or method == ["direct", "trace"]:
			assert method == "trace", "Invalid method specified"
			from primate.functional import hutch

			return hutch(LA, fun=fun, **kwargs)
		else:
			assert isinstance(fun, Callable), "'fun' must be callable"
			ew = np.linalg.eigvalsh(LA.tosparse().todense())
			return np.sum(fun(ew))

	def query(
		self,
		p: int,
		a: float,
		b: float,
		c: float = None,
		d: float = None,
		delta: float = 1e-12,
		summands: bool = False,
		**kwargs,
	) -> float:
		"""Queries the number of persistent pairs from Hp that intersect the box (a,b] x (c,d].

		Note that if multiple pairs lie exactly on the boundary of the box, only a fraction of them will be reported.
		"""
		q = p + 1
		if (c is None and d is None) or (c == -np.inf and d == np.inf):
			terms = [0] * 4
			terms[0] = np.sum(self._weights[p] <= a)
			terms[1] = self.rank(p, -np.inf, a, **kwargs)
			terms[2] = self.rank(q, -np.inf, b, **kwargs)
			terms[3] = self.rank(q, a + delta, b, **kwargs)
			return sum(s * t for s, t in zip([+1, -1, -1, +1], terms)) if not (summands) else terms
			# raise NotImplementedError("not implemented yet")
		else:
			pairs = [(b + delta, c), (a + delta, c), (b + delta, d), (a + delta, d)]
			# pattern = [(1,1),(1,0),(0,1),(0,0)]
			# terms = [self.rank(q, i+x*delta, j-y*delta, **kwargs) for cc, (x,y) in enumerate(pattern)]
			terms = [self.rank(q, i, j, **kwargs) for i, j in pairs]
			return sum(s * t for s, t in zip([+1, -1, -1, +1], terms)) if not (summands) else terms

	def query_dim(
		self,
		p: int,
		a: float,
		b: float,
		c: float = None,
		d: float = None,
		delta: float = 1e-12,
		summands: bool = False,
		**kwargs,
	) -> int:
		"""Queries the number of persistent pairs from Hp that intersect the box (a,b] x (c,d].

		Note that if multiple pairs lie exactly on the boundary of the box, only a fraction of them will be reported.
		"""
		return self.query(p, a, b, c, d, delta, summands, **kwargs)

	def query_spectral(
		self,
		p: int,
		a: float,
		b: float,
		c: float = None,
		d: float = None,
		summands: bool = False,
		fun: Callable = np.sign,
		**kwargs,
	):
		"""Queries the dimension of the persistent homology class H_p(a,b,c,d)."""
		q = p + 1
		if (c is None and d is None) or (c == -np.inf and d == np.inf):
			terms = [0] * 4
			terms[0] = np.sum(fun(self._weights[p][self._weights[p] <= a]))
			terms[1] = self.spectral_sum(p, 0, a, fun, **kwargs)
			terms[2] = self.spectral_sum(q, 0, b, fun, **kwargs)
			terms[3] = self.spectral_sum(q, a, b, fun, **kwargs)
			return sum(s * t for s, t in zip([+1, -1, -1, +1], terms)) if not (summands) else terms
			# raise NotImplementedError("not implemented yet")
		else:
			pairs = [(b, c), (a, c), (b, d), (a, d)]
			terms = [self.spectral_sum(q, i, j, fun, **kwargs) for cc, (i, j) in enumerate(pairs)]
			return sum(s * t for s, t in zip([+1, -1, -1, +1], terms)) if not (summands) else terms

	def _index_weights(self):
		D = len(self._weights)
		weights = np.hstack([self._weights[q] for q in range(D)])
		ranks = np.hstack([self._simplices[q] for q in range(D)])
		dims = np.hstack([np.repeat(q, len(self._simplices[q])) for q in range(D)])
		wrd = np.array(list(zip(weights, ranks, dims)), dtype=[("weight", "f4"), ("rank", "i8"), ("dim", "i4")])
		# wrd['rank'] = -wrd['rank']
		wrd_ranking = np.argsort(np.argsort(wrd, order=("weight", "dim", "rank")))  #  + 1
		index_weights = {q: wrd_ranking[dims == q] for q in range(D)}
		return index_weights, wrd

	## Queries the persistent pairs via a logarithmic number of rank computations on the index persistence plane
	## Iteratively re-weights the underlying linear operator
	def query_pairs(
		self, p: int, a: float, b: float, c: float, d: float, delta: float = 1e-15, simplex_pairs: bool = False, **kwargs
	):
		"""Queries the persistent pairs via a logarithmic number of rank computations on the index persistence plane."""
		assert (
			a < b and b <= c and c < d
		), (
			f"Invalid box ({a},{b}]x({c},{d}] given; must satisfy a < b <= c < d!"
		)  # see eq. 2.1 in the persistent measure paper
		dgm_dtype = (
			[("birth", "f4"), ("death", "f4")]
			if not simplex_pairs
			else [("birth", "f4"), ("death", "f4"), ("creator", "i8"), ("destroyer", "i8")]
		)
		kwargs["method"] = kwargs.get("method", "cholesky")
		verbose: bool = kwargs.pop("verbose", False)

		## intiial check
		if self.query(p, a, b, c, d, **kwargs) == 0:
			return np.empty(shape=(0,), dtype=dgm_dtype)

		weights_backup = self._weights.copy()
		self._weights, wrd = self._index_weights()

		## The real-valued weights, as a vector
		rw = np.hstack([weights_backup[q] for q in range(len(self._weights))])
		# iw = np.hstack([self._weights[q] for q in range(len(self._weights))])
		# np.argsort(np.argsort(rw))

		## Translate the query into a *valid* query on the index persistence plane
		## NOTE: The offsets are needed because we cannot handle degenerate boxes on index-persistence
		## We use sum instead of searchsorted because the weights are not ordered!
		bi = (
			np.sum(b >= rw) - 1
		)  # np.sum(b >= rw) - 1 ## the left is a the tightest inclusive query (rounding down), the right is one more
		di = np.sum(d >= rw) - 1  # np.sum(d >= rw) -1 ## The right works on the index persistence plane
		ai = max(np.sum(a >= rw) - 1, 0)  # max(min(np.sum(rw <= a) - 1, bi-1), 0)
		ci = max(np.sum(c >= rw) - 1, bi)  # max(min(np.sum(rw <= c) - 1, di-1), bi)

		## Checks
		rw_sorted = np.sort(rw)
		# assert b <= rw_sorted[bi], f"Index mapped b={b} must snap right to {rw_sorted[bi]}"
		# if bi > 0:
		#   assert rw_sorted[bi-1] <= b, f"Index mapped b={b} must be larger than {rw_sorted[bi-1]}"

		# assert rw_sorted[ai] <= a, f"Index mapped a={a} must snap left to {rw_sorted[ai]}"
		# if bi > 0:
		#   assert rw_sorted[bi-1] <= b, f"Index mapped b={b} must be larger than {rw_sorted[bi-1]}"
		# assert rw_sorted[di] <= d, f"Index mapped b={b} must snap left to {rw_sorted[bi]}"
		# # if di < (len(rw_sorted) - 1):
		# #   assert d < rw_sorted[di+1], f"Index mapped d={d} must be less than outer index {rw_sorted[di+1]}"
		# assert True if di == 0 else rw_sorted[di-1] < d and d <= rw_sorted[di], "Index mapped b must include the b interval"
		if verbose:
			print(f"Index translated box: [{ai}, {bi}] x [{ci}, {di}]")
			print(
				f"Function translated box: [{rw_sorted[ai]:.3f}, {rw_sorted[bi]:.3f}] x [{rw_sorted[ci]:.3f}, {rw_sorted[di]:.3f}]"
			)

		## Do the divide and conquer pair search
		if verbose:
			print(f"Method = {kwargs['method']}")
		pairs = points_in_box(ai, bi, ci, di, lambda i, j, k, l: self.query(p, i, j, k, l, **kwargs), verbose)
		if verbose:
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
		return np.array([tuple(pair) for pair in p_dgm], dtype=dgm_dtype)

	def query_representatives(self, p: int, pair: tuple, delta: float = 1e-15, simplex_pairs: bool = False, **kwargs):
		"""Given a pair of simplices which is known to be a p-dimensional persistent pair, this function finds a representative p-cycle."""
		raise NotImplementedError("Not implemented yet")
		# A = D[:ess_index, :ess_index]             ## All simplices preceeding creator
		# b = D[:ess_index, [ess_index]].todense()  ## Boundary chain of creator simplex

		# ## Since A is singular, it cannot be factored directly, so solve least-squares min ||Ax - b||_2
		# x, istop, itn, r1norm = lsqr(A.tocsc(), b)[:4]
		# cycle_rep_indices = np.flatnonzero(x != 0)
		# rep_cycle_simplices = [sx.Simplex(K[i]) for i in cycle_rep_indices]
		pass

	def operators(
		self, p: int, a: float, b: float, c: float = None, d: float = None, delta: float = 1e-12, **kwargs
	) -> float:
		"""Constructs operators associated with a given query (a,b] x (c,d] in B_p."""
		f = p - 1
		if (c is None and d is None) or (c == -np.inf and d == np.inf):
			from scipy.sparse import diags

			op1 = diags(np.where(self._weights[p] <= a, self._weights[p], 0.0))
			op2 = self.lower_left(f, -np.inf, a)
			op3 = self.lower_left(p, -np.inf, b)
			op4 = self.lower_left(p, a + delta, b)
			return (op1, op2, op3, op4)
			# raise NotImplementedError("not implemented yet")
		else:
			pairs = [(b + delta, c), (a + delta, c), (b + delta, d), (a + delta, d)]
			ops = [self.lower_left(p, i, j) for i, j in pairs]
			return ops
