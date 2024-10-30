import numpy as np
from typing import Callable
from collections import deque


def midpoint(i: int, j: int) -> int:
	return (i + j) // 2


def _generate_boxes(i: int, j: int, res: dict = {}, index: int = 0):
	"""Recursively generates boxes in DFS fashion"""
	# r = (i, (i+j) // 2, (i+j) // 2 + 1, j) # Chens version
	r = (i, (i + j) // 2, (i + j) // 2, j)  # Modified version
	res[index] = r
	if abs(i - j) <= 1:
		return
	else:
		_generate_boxes(i, (i + j) // 2, res, 2 * index + 1)
		_generate_boxes((i + j) // 2, j, res, 2 * index + 2)


def invert_dfs_box(n: int, index: int):
	"""Returns the box (i, mid-left, mid-right, j) for a given DFS index."""
	left, right = 0, n - 1
	path = []
	while index > 0:
		path.append(index % 2)  # Track left (1) or right (0)
		index = (index - 1) // 2  # Move to parent
	path.reverse()  # Start from root downwards
	while path:
		mid = (left + right) // 2
		if path.pop(0) == 0:  # Right child
			left = mid
		else:  # Left child
			right = mid
	mid = (left + right) // 2
	return (left, mid, mid, right)


def _generate_bfs(i: int, j: int):
	"""Generates a level-order indexed set of boxes on the bisection in via iterative BFS."""
	res, d, index = {}, deque(), 0
	d.appendleft((i, j))
	while len(d) > 0:
		i, j = d.pop()
		if abs(i - j) <= 1:
			continue
		else:
			# r = (i, (i+j) // 2, (i+j) // 2 + 1, j) # Chens version
			r = (i, (i + j) // 2, (i + j) // 2, j)  # modified version
			res[index] = r
			d.appendleft((i, (i + j) // 2))
			d.appendleft(((i + j) // 2, j))
			index += 1
	return res


def bisection_tree(
	i1, i2, j1, j2, mu: int, query_fun: Callable, creators: dict = {}, verbose: bool = False, validate: bool = True
):
	if verbose:
		print(f"({i1},{i2},{j1},{j2}) = {mu}")
	assert mu >= 0, f"Invalid multiplicity query: multiplicity {mu} at box [{i1},{i2}] x [{j1},{j2}] must be non-negative"
	if mu == 0:
		return
	# elif i1 == i2 and mu == 1:
	elif i2 - i1 <= 1 and mu == 1:
		if verbose:
			print(f"Creator found at index {i2} with destroyer in [{j1}, {j2}]")
		creators[i2] = (j1, j2)
	else:
		assert i1 < i2, f"Invalid traversal: i1 >= i2 ({i1}, {i2})"
		mid = midpoint(i1, i2)
		mu_L = query_fun(i1, mid, j1, j2)
		mu_R = mu - mu_L
		mu_R_test = query_fun(mid, i2, j1, j2) if validate else mu_R
		if mu_R != mu_R_test:
			## Invariant to keep: mu([i1, mid] x [j1, j2]) + mu([mid,i2] x [j1,j2]) = mu([i1,i2,j1,j2])
			msg = f"Invalid query oracle: mu={mu} (veri: {query_fun(i1,i2,j1,j2)}) on [{i1},{i2}]x[{j1},{j2}] should be L:{mu_L} from [{i1},{mid}]x[{j1},{j2}] + R:{mu_R} from [{mid},{i2}]x[{j1},{j2}] (got L:{mu_L} R:{mu_R_test})"
			raise ValueError(msg)
		bisection_tree(i1, mid, j1, j2, mu_L, query_fun, creators, verbose)
		bisection_tree(mid, i2, j1, j2, mu_R, query_fun, creators, verbose)
		# bisection_tree(mid+1, i2, j1, j2, mu_R, query_fun, creators)
	return creators


# def find_negative(b: int, lc: int, rc: int, query_fun: Callable):
#   """Finds the unique j \in [j1,j2] satisfying query_fun(b,b,j,j) != 0 via binary search"""
#   ii = b
#   mu_j = query_fun(ii, ii, lc, rc)
#   if mu_j == 0:
#     return ii
#   while lc != rc:
#     # print(f"l, r: ({l}, {r})")
#     mu_L = query_fun(ii, ii, lc, midpoint(lc,rc))
#     lc, rc = (lc, (lc + rc) // 2) if mu_L != 0 else ((lc + rc) // 2 + 1, rc)
#   return lc


def find_negative(i: int, lc: int, rc: int, query_fun: Callable, verbose: bool = False):
	"""Finds the unique j \in (j-1,j] satisfying query_fun(i-1,i,j-1,j) = 1 via binary search"""
	mu_j = query_fun(i - 1, i, lc, rc)  ## Should always be 1
	if mu_j == 0:
		return -1
	while (rc - lc) > 1:
		if verbose:
			print(f"{i}: j \in ({lc}, {rc}]")
		mu_L = query_fun(i - 1, i, lc, midpoint(lc, rc))
		# lc, rc = (lc, (lc + rc) // 2) if mu_L != 0 else ((lc + rc) // 2 + 1, rc)
		lc, rc = (lc, (lc + rc) // 2) if mu_L != 0 else ((lc + rc) // 2, rc)
	return rc


## Given a black-box oracle function query: tuple -> int that accepts four integers in [0, N-1]
## and returns an integer indicating how many points lie in a given box [i,j] x [k,l] of a
## index-filtered persistence diagram of a simplicial complex K, this function finds the
## persistent pairs of those pairs
def points_in_box(i: int, j: int, k: int, l: int, query: Callable[tuple, int], verbose: bool = False) -> np.ndarray:
	mu_init = query(i, j, k, l)
	positive = {}
	bisection_tree(i, j, k, l, mu_init, query, positive, verbose)
	pairs = {b: find_negative(b, j1, j2, query, verbose) for b, (j1, j2) in positive.items()}
	return pairs


# def ph_pairs(K: sx.ComplexLike, p: int, query: Callable):
#   m: int = len(K)
#   boxes = {}
#   _generate_bfs(0, m, boxes) ## todo: remove and replace with log(n) indexing function
#   a, b = 0, len(K)
#   # query(a, midpoint(a,b), midpoint(a,b), b)

# def _index_persistence(K: sx.FiltrationLike, **kwargs):
#   from pbsig.persistence import ph
#   K = sx.RankFiltration(K)
#   K.order = 'reverse colex'
#   K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
#   dgms_index = ph(K_index, simplex_pairs=True)
#   return dgms_index

## TODO:
# 1. Write Points-in-box function (bisection tree + destroyer)
# 2. Write code to enumerate all B* boxes in order of area + left-to-right
# 3. Compute all pairs
# 4. Determine position needed to compute all pairs up to gamma-persistence
# 5. Biject it all back to the function domain
# 6. Interface it up
# 7. Test test test
