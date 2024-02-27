import numpy as np
import splex as sx
from typing import Callable
from collections import deque

def midpoint(i: int, j: int) -> int: 
  return (i + j) // 2

def _generate_boxes(i: int, j: int, res: dict = {}, index: int = 0):
  """Recursively generates the """
  r = (i, (i+j) // 2, (i+j) // 2 + 1, j)
  res[index] = r
  if abs(i-j) <= 1: 
    return  
  else:
    _generate_boxes(i, (i+j) // 2, res, 2*index + 1)
    _generate_boxes((i+j) // 2, j, res, 2*index + 2)

def _generate_bfs(i: int, j: int):
  res, d, index = {}, deque(), 0
  d.appendleft((i,j))
  while len(d) > 0:
    i,j = d.pop()
    if abs(i-j) <= 1: 
      continue
    else:
      r = (i, (i+j) // 2, (i+j) // 2 + 1, j)
      res[index] = r
      d.appendleft((i, (i+j) // 2))
      d.appendleft(((i+j) // 2, j))
      index += 1
  return res
 

## Doesn't quite work yet 
def get_index(k: int, i: int, j: int):
  # assert i <= k and 
  indices = [k]
  while k > 0:
    k = (k - 1) // 2
    indices.append(k)
  indices = np.flip(indices)
  for l in indices[1:]: 
    if l % 2 == 0:
      i,j = (i+j) // 2, j
    else:
      i,j = i, (i+j) // 2
  return i,j


def bisection_tree(i1, i2, j1, j2, mu: int, query_fun: Callable, creators: dict = {}, verbose: bool = False):
  if verbose: 
    print(f"({i1},{i2},{j1},{j2}) = {mu}")
  if mu == 0: 
    return
  elif i1 == i2 and mu == 1:
    if verbose:
      print(f"Creator found at index {i1} with destroyer in [{j1}, {j2}]") 
    creators[i1] = (j1, j2)
  else:
    mid = midpoint(i1, i2)
    mu_L = query_fun(i1, mid, j1, j2)
    mu_R = mu - mu_L 
    mu_R_test = query_fun(mid+1, i2, j1, j2)
    if mu_R != mu_R_test:
      msg = f"Invalid query oracle: mu={mu} on [{i1},{i2}]x[{j1},{j2}] should be L:{mu_L} from [{i1},{mid}]x[{j1},{j2}] + R:{mu_R} from [{mid+1},{i2}]x[{j1},{j2}] (got L:{mu_L} R:{mu_R_test})"
      raise ValueError(msg)
    bisection_tree(i1, mid, j1, j2, mu_L, query_fun, creators)
    bisection_tree(mid+1, i2, j1, j2, mu_R, query_fun, creators)
  return creators

def find_negative(b: int, lc: int, rc: int, query_fun: Callable):
  """Finds the unique j \in [j1,j2] satisfying query_fun(b,b,j,j) != 0 via binary search"""
  ii = b 
  mu_j = query_fun(ii, ii, lc, rc)
  if mu_j == 0: 
    return ii
  while lc != rc:
    # print(f"l, r: ({l}, {r})")
    mu_L = query_fun(ii, ii, lc, midpoint(lc,rc))
    lc, rc = (lc, (lc + rc) // 2) if mu_L != 0 else ((lc + rc) // 2 + 1, rc)
  return lc 

## Given a black-box oracle function query: tuple -> int that accepts four integers in [0, N-1]
## and returns an integer indicating how many points lie in a given box [i,j] x [k,l] of a 
## index-filtered persistence diagram of a simplicial complex K, this function finds the 
## persistent pairs of those pairs
def points_in_box(i: int, j: int, k: int, l: int, query: Callable[tuple, int]) -> np.ndarray:
  mu_init = query(i, j, k, l)
  positive = {}
  bisection_tree(i, j, k, l, mu_init, query, positive)
  pairs = { c : find_negative(c, j1, j2, query) for c, (j1, j2) in positive.items() }
  return pairs

def ph_pairs(K: sx.ComplexLike, p: int, query: Callable):
  m: int = len(K)
  boxes = {}
  _generate_bfs(0, m, boxes) ## todo: remove and replace with log(n) indexing function
  a, b = 0, len(K)
  query(a, midpoint(a,b), midpoint(a,b)+1, b)

## TODO: 
# 1. Write Points-in-box function (bisection tree + destroyer)
# 2. Write code to enumerate all B* boxes in order of area + left-to-right
# 3. Compute all pairs 
# 4. Determine position needed to compute all pairs up to gamma-persistence
# 5. Biject it all back to the function domain
# 6. Interface it up 
# 7. Test test test