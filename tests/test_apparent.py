import numpy as np 
import spirit
from scipy.spatial.distance import pdist, squareform, cdist
from math import comb
from combin import rank_to_comb, comb_to_rank
import splex as sx
from spirit import apparent_pairs

# dx = np.array([3,4,5,5,4,3])
# S = sx.rips_complex(dx, p=3)
# for i in range(5):
#   RM = apparent_pairs.rip_mod.ripser(dx, 2, 2*sx.enclosing_radius(dx))
#   t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n=sx.card(S,0))
#   t_diams = sx.flag_filter(dx)(sx.faces(S,2))
#   e_ranks = comb_to_rank(sx.faces(S, 1), order='colex', n=sx.card(S,0))
#   e_diams = sx.flag_filter(dx)(sx.faces(S,1))
  
#   ## Test apparent facets call
#   APs = RM.apparent_facets(t_ranks, t_diams, 2)
#   assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
#   e_verts = rank_to_comb(APs[APs != -1], k=2, order='colex', n = sx.card(S,0))
#   assert np.allclose(np.ravel(e_verts), np.array([0,3,1,2]))

def test_rank_ll():
  X = np.random.uniform(size=(10,2))
  K = sx.rips_filtration(X, p = 2, form = 'rank')
  D2 = sx.boundary_matrix(K, p = 2)

  ## Collect the apparent pairs  
  S = sx.rips_complex(X, p = 2)
  f = sx.flag_filter(pdist(X))
  APs = apparent_pairs.apparent_pairs(S, f, p = 1)
  assert True

def test_paper_example():
  dx = np.array([3,4,5,5,4,3])
  S = sx.rips_complex(dx, p=3)
  DX = squareform(dx)
  R = apparent_pairs.rip_mod.ripser_lower(dx, 2, 2*sx.enclosing_radius(dx))
  index_21 = comb_to_rank([2,1], order='colex', n=sx.card(S,0))
  diam_21 = DX[2,1]
  diam_321 = np.max([DX[3,2], DX[1,2], DX[1,3]])
  index_321 = comb_to_rank([3,2,1], order='colex', n=sx.card(S,0))
  assert np.isclose(diam_21, diam_321)
  
  ## ahhh duality! 
  c = R._apparent_cofacet(index_321, diam_321, 1)
  assert c == index_21
  c = R._apparent_facet(index_21, diam_21, 2)
  assert c == index_321


def test_paper_example2():
  dx = np.array([3,4,5,5,4,3])
  S = sx.rips_complex(dx, p=3)
  for i in range(5):
    RM = apparent_pairs.rip_mod.ripser_lower(dx, 2, 2*sx.enclosing_radius(dx))
    t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n=sx.card(S,0))
    t_diams = sx.flag_filter(dx)(sx.faces(S,2))
    e_ranks = comb_to_rank(sx.faces(S, 1), order='colex', n=sx.card(S,0))
    e_diams = sx.flag_filter(dx)(sx.faces(S,1))
    
    ## Test apparent facets call
    APs = RM.apparent_facets(t_ranks, t_diams, 2)
    assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
    e_verts = rank_to_comb(APs[APs != -1], k=2, order='colex', n = sx.card(S,0))
    assert np.allclose(np.ravel(e_verts), np.array([0,3,1,2]))

    ## Test apparent cofacets call
    APs = RM.apparent_cofacets(e_ranks, e_diams, 1)
    assert np.all(np.flatnonzero(APs != -1) == np.array([2,3]))
    t_verts = rank_to_comb(APs[APs != -1], k=3, order='colex', n = sx.card(S,0))
    assert np.allclose(np.ravel(t_verts), np.array([0,2,3,1,2,3]))

def test_apparent_facet():
  from scipy.spatial.distance import pdist
  from spirit.apparent_pairs import clique_mod
  X = np.random.uniform(size=(10,2))
  M = clique_mod.MetricSpace(10,2)
  r = comb_to_rank([8,5,3], order='colex', n = 10)
  M.weights = pdist(X)
  # M.boundary(r, 2)
  # M.coboundary(r, 2)
  assert np.isclose(M.simplex_weight(r, 2), np.max(pdist(X[[3,5,8]])))

  M.apparent_facet(r, 2)
  M.simplex_weight(33,1)
  assert np.all(rank_to_comb([33], n=M.n, k=2, order='colex') == np.array([5,8]))

  from itertools import combinations
  edges = np.array([comb_to_rank(c, k=2, order='colex', n=M.n) for c in combinations(range(M.n), 2)])
  triangles = np.array([comb_to_rank(c, k=3, order='colex', n=M.n) for c in combinations(range(M.n), 3)])
  
  facet_ranks = np.array([M.apparent_facet(r, 2) for r in triangles])
  cofacet_ranks = np.array([M.apparent_cofacet(e,1) for e in edges])

  for tr in triangles: 
    facet_ranks[triangles == 84]
    edges[cofacet_ranks == 84]

  pass

def test_apparent_pairs():
  f_vals = np.array([0,0,0,0,3,3,4,4,5,5,5,5,5,5,5])
  s_vals = [[3],[2],[1],[0],[3,2],[1,0],[3,1],[2,0],[3,0],[2,1],[3,2,1],[3,2,0],[3,1,0],[2,1,0],[3,2,1,0]]
  K = sx.RankFiltration(zip(f_vals, s_vals))
  K.order = 'reverse colex'
  
  from spirit.apparent_pairs import clique_mod
  M = clique_mod.MetricSpace(sx.card(K, 0), 3)
  M.weights = np.array([3,4,5,5,4,3])
  # M.simplex_weight(r, 2)

  er = comb_to_rank(sx.faces(K,1), order='colex', n=sx.card(K,0))
  tr = comb_to_rank(sx.faces(K,2), order='colex', n=sx.card(K,0))
  qr = comb_to_rank(sx.faces(K,3), order='colex', n=sx.card(K,0))

  #zero_er = np.array([M.zero_facet(r, 2) for r in tr])
  #zero_tr = np.array([M.zero_cofacet(r, 1) for r in er])

  # [M.apparent_zero_facet(r, 2) for r in tr]
  ## Ensure we can deflate the nullspace using apparent pairs
  ap_tr = np.array([M.apparent_zero_cofacet(r, 1) for r in er])
  D1 = sx.boundary_matrix(K, 1).todense()
  D1_rank = np.linalg.matrix_rank(D1)
  assert np.linalg.matrix_rank(D1[:,ap_tr == -1]) == D1_rank

  ap_tr = np.array([M.apparent_zero_cofacet(r, 2) for r in tr])
  D2 = sx.boundary_matrix(K, 2).todense()
  D2_rank = np.linalg.matrix_rank(D2)
  assert np.linalg.matrix_rank(D2[:,ap_tr == -1]) == D2_rank

def test_med_rips():
  X = np.random.uniform(size=(20, 2))
  S = sx.rips_complex(X, p=2)
  f = sx.flag_filter(pdist(X))
  # # K = sx.rips_filtration(X, p=2)

  t_diams = f(sx.faces(S,2))
  t_ranks = comb_to_rank(sx.faces(S, 2), order='colex', n = sx.card(S, 0))

  DX = squareform(pdist(X))
  # DX = np.ravel(np.tril(DX))[np.ravel(np.tril(DX)) != 0]
  DX = np.ravel(np.triu(DX))[np.ravel(np.triu(DX)) != 0]

  RM = apparent_pairs.rip_mod.ripser_lower(DX, 2, 4*sx.enclosing_radius(X))
  RM.apparent_facets(t_ranks, t_diams, 2)

  RM._apparent_cofacet(t_ranks[1], t_diams[1], 2)




def test_apparent_facet():
  from combin import rank_to_comb, comb_to_rank
  from spirit.apparent_pairs import clique_mod
  from math import comb
  
  # rank_to_comb(0, k=2, n=10)
  # r = comb_to_rank([5,3,0], order='colex', n=7)
  r = comb(5,3) + comb(3,2) + comb(0,1)
  v = [5,3,0]
  
  idx_below = r
  idx_above = 0
  j = n - 1
  k = 2 + 1
  ii = 0

  while (j >= k and comb(j, k) > idx_below):
    print(f"j = {j}, k = {k}, idx_above={idx_above}, idx_below={idx_below}")
    while (comb(j, k) <= idx_below):
      idx_below -= comb(j, k)
      idx_above += comb(j, k + 1)
      j = j - 1
      k = k - 1
      print(f"idx_below={idx_below}, idx_above={idx_above}, j = {j}, k = {k}")
    cofacet_index = idx_above + comb(j, k + 1) + idx_below

    print(f"cofacet: {cofacet_index}")
    j = j - 1
    idx_below = idx_below - comb(v[ii], k)
    ii += 1
    # idx_below = cofacet_index

  ## Test: (8, 5, 1) w/ n = 13

  comb_to_rank([12,8,5,1], order='colex', n = 13)
  comb_to_rank([11,8,5,1], order='colex', n = 13)
  comb_to_rank([10,8,5,1], order='colex', n = 13)
  comb_to_rank([9,8,5,1], order='colex', n = 13)

  from scipy.special import comb
  # n = 13
  # n, d = 7, 2
  # sigma = [5,3,0]
  n, d = 7, 2
  sigma = [6,4,1]
  r = comb_to_rank(sigma, order='colex', n = n)
  idx_above, idx_below = 0, r
  # vertices = np.flip([0] + sigma)
  k = d + 1
  v = apparent_pairs.clique_mod.get_max_vertex(r, d+1, n) - 1 # max vertex
  
  ## start with the right 'below' sum ?
  # idx_below -= comb(v, k-1, exact=True)
  # idx_below += 

  for j in reversed(range(0, n)):
    if j in sigma: 
      continue
    k = (d + 1) - np.searchsorted(-np.array(sigma), -j)
    # while v > j:
    #   k -= 1
    # if k == (len(vertices) - 1):
    #   vertices[k] = j
    # else:
    #   vertices[[k,k+1]] = vertices[[k+1,k]]
    #   vertices[k] = j
    num = 0 if k >= (d+1) else v
    idx_above += comb(num, k+2, exact=True)
    cofacet_rank = idx_above + comb(j, k+1, exact=True) + idx_below
    print(f"{idx_above}, {comb(j, k+1, exact=True)}, {idx_below}")
    print(f"cofacet rank: {cofacet_rank}, cofacet: {rank_to_comb(cofacet_rank, k=d+2, order='colex', n=n)}")
    v = apparent_pairs.clique_mod.get_max_vertex(idx_below, k, n) - 1 # max vertex
    idx_below -= comb(v, k, exact=True) ## post? 

def test_colex_equation():
  n, d = 7, 2
  sigma = np.flip([5,3,0])
  r = comb_to_rank(sigma, order='colex', n = n)
  V = [6,4,2,1]
  for j in V:
    k = np.searchsorted(sigma, j)
    idx_above = sum([comb(sigma[l], l+2, exact=True) for l in range(k,d+1)])
    idx_middle = comb(j, k+1, exact=True)
    idx_below = sum([comb(sigma[l], l+1, exact=True) for l in range(0,k)])
    cofacet_rank = idx_above + idx_middle + idx_below
    cofacet = rank_to_comb(cofacet_rank, order='colex', k=4)
    print(f"{tuple(reversed(cofacet))} <=> {idx_above} + {idx_middle} + {idx_below} = {cofacet_rank} (k={k})")
    assert tuple(np.sort(np.append(sigma, [j]))) == cofacet

def test_colex_equation2():
  n, d = 13, 2
  sigma = np.flip([8, 5, 1])
  r = comb_to_rank(sigma, order='colex', n = n)
  V = reversed(np.setdiff1d(np.arange(n), sigma))
  for j in V:
    k = np.searchsorted(sigma, j)
    idx_above = sum([comb(sigma[l], l+2, exact=True) for l in range(k,d+1)])
    idx_middle = comb(j, k+1, exact=True)
    idx_below = sum([comb(sigma[l], l+1, exact=True) for l in range(0,k)])
    cofacet_rank = idx_above + idx_middle + idx_below
    cofacet = rank_to_comb(cofacet_rank, order='colex', k=4)
    print(f"{tuple(reversed(cofacet))} <=> {idx_above} + {idx_middle} + {idx_below} = {cofacet_rank} (k={k})")
    assert tuple(np.sort(np.append(sigma, [j]))) == cofacet

def test_colex_update_fast():
  from scipy.special import comb
  n, d = 13, 2
  # sigma = np.flip([0,5,3,0])
  sigma = np.flip([0,8,5,1])
  r = comb_to_rank(sigma[:(d+1)], order='colex', n = n)
  V = list(reversed(np.setdiff1d(np.arange(n), sigma[:(d+1)])))
  idx_above = 0
  idx_below = r
  k = d + 1
  for j in V:
    k = int(sigma[0] < j) + int(sigma[1] < j) + int(sigma[2] < j)
    assert k == np.searchsorted(sigma[:(d+1)], j)
    while comb(j, k, exact=True) <= idx_below:
      idx_above += comb(sigma[k], k+2, exact=True)
      idx_below -= comb(sigma[k], k+1, exact=True)
    idx_middle = comb(j, k+1, exact=True)
    cofacet_rank = idx_above + idx_middle + idx_below
    
    ## Validate 
    cofacet = rank_to_comb(cofacet_rank, order='colex', k=4)
    print(f"{tuple(reversed(cofacet))} <=> {idx_above} + {idx_middle} + {idx_below} = {cofacet_rank} (k={k})")
    assert tuple(np.sort(np.append(sigma[:-1], [j]))) == cofacet


def test_colex_update_fast2():
  assert True 
  pass 
  from scipy.special import comb
  n, d = 7, 2
  # s = np.sort(np.random.choice(range(n), size=d+1, replace=False))
  s = [3,4,6]
  # s = [0,3,5]
  # sigma = np.flip([0,14,8,5,1])
  sigma = np.append(s, [0])
  r = comb_to_rank(sigma[:(d+1)], order='colex', n = n)
  V = list(reversed(np.setdiff1d(np.arange(n), sigma[:(d+1)])))
  idx_above = 0
  idx_below = r
  k = d + 1
  print(f"sigma: {s}, n = {n}")
  for j in V:
    k = int(sigma[0] < j) + int(sigma[1] < j) + int(sigma[2] < j)
    assert k == np.searchsorted(sigma[:(d+1)], j)
    while comb(j, k, exact=True) <= idx_below:
      idx_above += comb(sigma[k], k+2, exact=True)
      idx_below -= comb(sigma[k], k+1, exact=True)
    idx_middle = comb(j, k+1, exact=True)
    cofacet_rank = idx_above + idx_middle + idx_below
    
    ## Validate 
    cofacet = rank_to_comb(cofacet_rank, order='colex', k=d+2)
    print(f"{tuple(reversed(cofacet))} <=> {idx_above} + {idx_middle} + {idx_below} = {cofacet_rank} (k={k})")
    true_cofacet = tuple(np.sort(np.append(sigma[:-1], [j])))
    assert true_cofacet == cofacet, "Cofacets don't match!"


# def test_max_vertex():
#   n, d = 7, 2
#   sigma = [5,3,0]
#   r = comb_to_rank(sigma, order='colex', n = n)
#   v = apparent_pairs.clique_mod.get_max_vertex(r, d+1, n) - 1 # max vertex
#   # Binary searches for the value K satisfying choose(K-1, m) <= r < choose(K, m) 
#   assert j == 6
  
## Indeed, it's unclear how ripser code works at all!
def test_ripser_coboundary():
  from scipy.special import comb
  n, d = 30, 3
  # sigma = [5,3,0]
  sigma = np.sort(np.random.choice(range(n), size=d+1, replace=False))
  r = comb_to_rank(sigma, order='colex', n = n)
  idx_above, idx_below = 0, r
  j = n - 1
  k = d + 1 
  print(f"sigma: {sigma}, n = {n}")
  while j >= k:
    while (comb(j, k, exact=True) <= idx_below):
      idx_below -= comb(j, k, exact=True)
      idx_above += comb(j, k + 1, exact=True)
      j = j - 1
      k = k - 1
      assert(k != -1)
    cofacet_index = idx_above + comb(j, k + 1) + idx_below
    true_cofacet = tuple(np.sort(np.append(sigma, [j]))) 
    test_cofacet = rank_to_comb(int(cofacet_index), order='colex', k=d+2)
    assert test_cofacet == true_cofacet
    j = j - 1
    print(f"{cofacet_index} -> {np.flip(test_cofacet)}")
    
def test_spirit_coboundary():
  from spirit.apparent_pairs import clique_mod
  n, d = 7, 2
  r = comb_to_rank([5,3,0], n=n, order='colex')
  cofacet_ranks = clique_mod.coboundary(r, d, n)
  assert tuple(cofacet_ranks) == (28, 12, 7, 6)

  ## Bigger example
  n, d = 30, 4
  sigma = np.sort(np.random.choice(range(n), size=d+1, replace=False))
  r = comb_to_rank(sigma, order='colex', n = n)
  cofacet_ranks_test = np.sort(clique_mod.coboundary(r, d, n))
  cofacet_ranks_true = np.sort(np.array([comb_to_rank(np.append(sigma, [j]), n=n, order='colex') for j in np.setdiff1d(np.arange(n), sigma)]))
  assert np.all(cofacet_ranks_true == cofacet_ranks_test)

def test_spirit_boundary():
  from spirit.apparent_pairs import clique_mod
  n, d = 7, 2
  r = comb_to_rank([5,3,0], n=n, order='colex')
  b = clique_mod.boundary(r, d, n)
  assert np.all(np.sort(b) == np.array([3,10,13]))

def test_boundary_ranks():
  from spirit.apparent_pairs import clique_mod
  from scipy.special import comb
  n, d = 7, 2
  r = comb_to_rank([5,3,0], n=n, order='colex')
  idx_below = r
  idx_above = 0
  j = n - 1
  for k in reversed(range(0, d+1)):
    j = clique_mod.get_max_vertex(idx_below, k + 1, j) - 1
    c = comb(j, k + 1, exact=True)
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += comb(j, k, exact=True)
    k -= 1
    print(f"{face_index} -> { rank_to_comb(face_index, k=d, n=n, order='colex')}")


def test_boundary_ranks():
  from spirit.apparent_pairs import clique_mod
  from scipy.special import comb
  n, d = 30, 4
  s = np.sort(np.random.choice(range(n), size=d+1, replace=False))
  r = comb_to_rank(s, n=n, order='colex')
  idx_below = r
  idx_above = 0
  j = n - 1
  print(s)
  for k in reversed(range(0, d+1)):
    j = clique_mod.get_max_vertex(idx_below, k + 1, j) - 1
    c = comb(j, k + 1, exact=True)
    face_index = idx_above - c + idx_below
    idx_below -= c
    idx_above += comb(j, k, exact=True)
    k -= 1
    print(f"{face_index} -> { rank_to_comb(face_index, k=d, n=n, order='colex')}, (j={j})")
