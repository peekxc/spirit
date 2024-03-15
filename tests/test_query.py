import numpy as np 
import splex as sx 
from spirit.query import bisection_tree, points_in_box, find_negative
from scipy.spatial.distance import pdist
from ripser import ripser
from pbsig.persistence import ph, generate_dgm, low_entry
from spirit.query import _generate_bfs, midpoint, bisection_tree, points_in_box
from spirit.apparent_pairs import SpectralRI
from itertools import chain
from combin import comb_to_rank

def index_pers_RI(K):
  N = len(K)
  RI = SpectralRI(N, max_dim=2)
  RI._simplices[0] = comb_to_rank(sx.faces(K,0), k=1, order='colex', n=N)
  RI._simplices[1] = comb_to_rank(sx.faces(K,1), k=2, order='colex', n=N)
  RI._simplices[2] = comb_to_rank(sx.faces(K,2), k=3, order='colex', n=N)
  RI._weights[0] = np.array([i for i,s in K if sx.dim(s) == 0]) # np.arange(sx.card(K,0))
  RI._weights[1] = np.array([i for i,s in K if sx.dim(s) == 1]) # np.arange(sx.card(K,1)) + max(RI._weights[0]) + 1
  RI._weights[2] = np.array([i for i,s in K if sx.dim(s) == 2]) # np.arange(sx.card(K,2)) + max(RI._weights[1]) + 1
  RI._status[0] = np.zeros(len(RI._simplices[0]))
  RI._status[1] = np.zeros(len(RI._simplices[1]))
  RI._status[2] = np.zeros(len(RI._simplices[2]))
  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)
  return RI

def rips_example(X, index: bool = False):
  S = sx.rips_complex(X, p=2)
  K = sx.RankFiltration(S, f=sx.flag_filter(pdist(X)))
  K.order = 'reverse colex'
  if index:
    K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
    dgms_index = ph(K_index, simplex_pairs=True)
    return K_index, dgms_index
  else:
    dgms = ph(K, simplex_pairs=True)
    return K, dgms

def index_pers_example():
  S = sx.RankComplex([[0,1,2], [3,4,5], [2,3], [1,4]])
  K = sx.RankFiltration(enumerate(chain(sx.faces(S, 0), sx.faces(S, 1), sx.faces(S, 2))))
  K.order = 'reverse colex'
  R, V = ph(K, output="RV")
  dgms_index = generate_dgm(K, R, simplex_pairs=True)
  return S, K, R, V, dgms_index

def show_pers(K, dgms, p: int = 0, index: bool = False):
  from pbsig.vis import figure_dgm, show, output_notebook
  output_notebook()
  p = figure_dgm(dgms[p])
  if index:
    p.xaxis[0].ticker = np.arange(len(K))
    p.yaxis[0].ticker = np.arange(len(K))
  show(p)

def sparse_str(D) -> str:
  return np.array2string(D, max_line_width=1024, separator=" ", prefix="", suffix="", formatter={'int' : lambda x: '1' if x == 1 else ' ' })

def test_pairing_uniqueness():
  S, K, R, V, dgms_index = index_pers_example()
  D = sx.boundary_matrix(K).todense() % 2
  R = R.todense().astype(int) % 2
  P = R.copy()
  for j, i in enumerate(low_entry(P)):
    P[:i,j] = 0
  print(sparse_str(D))
  print(sparse_str(R))
  print(sparse_str(P))

  ## Verify the rank formula from 3.5 in CTDA book
  RI = index_pers_RI(K)
  dims = np.array([sx.dim(s) for i,s in K])
  for (j, i), p in zip(enumerate(low_entry(R)), dims):
    if i != -1:
      ## (i,j) is a pivot <=> low(j) = i
      t1 = np.linalg.matrix_rank(R[i:,:(j+1)])
      t2 = np.linalg.matrix_rank(R[(i+1):,:(j+1)])
      t3 = np.linalg.matrix_rank(R[(i+1):,:j])
      t4 = np.linalg.matrix_rank(R[i:,:j])
      assert t1 - t2 + t3 - t4 == 1
      
      t1_test = np.linalg.matrix_rank(RI.lower_left(i+0, j+0, p=p).D.todense())
      t2_test = np.linalg.matrix_rank(RI.lower_left(i+1, j+0, p=p).D.todense())
      t3_test = np.linalg.matrix_rank(RI.lower_left(i+1, j-1, p=p).D.todense())
      t4_test = np.linalg.matrix_rank(RI.lower_left(i+0, j-1, p=p).D.todense())
      assert t1 == t1_test
      assert t2 == t2_test
      assert t3 == t3_test
      assert t4 == t4_test

def test_box_base_cases():
  ## Test all the interpretations of the boxes
  S, K, R, V, dgms_index = index_pers_example()
  # show_pers(K, dgms_index, 1)
  RI = index_pers_RI(K)
  assert RI.query(0, 0, 4, 4, 8) == 2 
  assert RI.query(0, 0, 1, 5, 6) == 1
  assert RI.query(0, 0, 2, 5, 6) == 1
  assert RI.query(0, 0, 1, 5, 7) == 1
  assert RI.query(0, 1, 2, 4, 6) == 0
  assert RI.query(0, 2, 3, 6, 8) == 0
  assert RI.query(0, 2, 3, 7, 8) == 0
  assert RI.query(0, 2, 7, 7, 8) == 0
  assert RI.query(0, 1, 3, 7, 8) == 0
  assert RI.query(0, 1, 2, 7, 7) == 0 
  assert RI.query(0, 2, 2, 7, 8) == 0
  assert RI.query(0, 2, 2, 5, 5) == 0
  assert RI.query(0, 2, 4, 9, 9) == 0
  assert RI.query(1, 7, 9, 13, 15) == 1
  assert RI.query(1, 7, 8, 13, 14) == 1
  assert RI.query(1, 8, 9, 14, 15) == 0

def test_bisection_simple():
  S, K, R, V, dgms_index = index_pers_example()
  RI = index_pers_RI(K)
  show_pers(K, dgms_index, 0)

  def query_oracle_truth(dgms_index, p: int = 0):
    def _query(i: int, j: int, k: int, l: int):
      in_birth = np.logical_and(i < dgms_index[p]['birth'] , dgms_index[p]['birth'] <= j)
      in_death = np.logical_and(k < dgms_index[p]['death'] , dgms_index[p]['death'] <= l)
      return np.sum(in_birth.view(bool) & in_death.view(bool))
    return _query

  def query_oracle(p: int = 0) -> int:
    def _query(i: int, j: int, k: int, l: int):
      return RI.query(p,i,j,k,l)
    return _query

  assert find_negative(1, 5, 8, query_oracle(0), verbose=False) == 6
  assert find_negative(2, 5, 8, query_oracle(0), verbose=False) == 7
  assert points_in_box(0,3,5,8, query=query_oracle(0)) == {1: 6, 2: 7}
  assert points_in_box(0,4,4,11, query=query_oracle(0)) == {1: 6, 2: 7, 3: 9, 4: 10}
  assert points_in_box(3,5,8,12, query=query_oracle(0)) == {4: 10, 5: 12}
  assert points_in_box(3,5,8,12, query=query_oracle(1)) == {}
  assert points_in_box(7,9,13,14, query=query_oracle(1)) == {8: 14}
  assert points_in_box(11,13,14,15, query=query_oracle(1)) == {13: 15}

  # show_pers(K, dgms_index, 1)
  assert points_in_box(3,5,8,12, query=query_oracle_truth(dgms_index, 1)) == {}
  assert points_in_box(7,9,13,14, query=query_oracle_truth(dgms_index, 1)) == {8: 14}
  assert points_in_box(11,13,14,15, query=query_oracle_truth(dgms_index, 1)) == {13: 15}

  ## Test we can recover all the points 
  assert [tuple(bc) for bc in RI.query_pairs(0, 0, 3, 5, 7)] == [(1,6), (2,7)]
  assert [tuple(bc) for bc in RI.query_pairs(0, 2, 4, 8, 10)] == [(3,9), (4,10)]
  assert [tuple(bc) for bc in RI.query_pairs(0, 4, 5, 9, 10)] == []
  assert [tuple(bc) for bc in RI.query_pairs(0, 3, 4, 9, 10)] == [(4,10)]

def test_bisection_oracle():
  for ii in range(5):
    np.random.seed(ii)
    N = 32
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)

    ## Start with a reverse colexicographically filtered complex with index filter values
    S = sx.rips_complex(X, p=2)
    K = sx.RankFiltration(S, f=sx.flag_filter(pdist(X)))
    K.order = 'reverse colex'
    K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
    dgms_index = ph(K_index, simplex_pairs=True)
    H1_b, H1_d = dgms_index[1]['birth'], dgms_index[1]['death']
    # show_pers(K, dgms_index, 1)

    def query_oracle(i: int, j: int, k: int, l: int) -> int:
      # in_birth = np.logical_and(i <= H1_b, H1_b <= j)
      # in_death = np.logical_and(k <= H1_d, H1_d <= l)
      in_birth = np.logical_and(i < H1_b, H1_b <= j)
      in_death = np.logical_and(k < H1_d, H1_d <= l)
      return np.sum(np.logical_and(in_birth, in_death))

    ## Generate all persistence pairs in the diagram
    from spirit.query import _generate_bfs
    all_boxes = _generate_bfs(0, len(K)-1)
    all_pairs = { k : points_in_box(*box, query_oracle, verbose=False) for k, box in all_boxes.items() }
    all_pairs = np.array([np.ravel(tuple(p.items())) for p in all_pairs.values() if len(p) > 0])
    all_pairs = all_pairs[np.lexsort(np.rot90(all_pairs)),:]
    H1_pairs = np.c_[dgms_index[1]['birth'], dgms_index[1]['death']]
    assert np.allclose(all_pairs - H1_pairs, 0)

def test_bisection_query():
  np.random.seed(1234)
  verbose = False
  N = 16
  theta = np.linspace(0, 2*np.pi, N, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)
  S = sx.rips_complex(X, p=2)
  K = sx.RankFiltration(S, f=sx.flag_filter(pdist(X)))
  K.order = 'reverse colex'
  K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
  dgms_index = ph(K_index, simplex_pairs=True)

  ## Start with index persistence, since this is like identity transform for query_pairs
  RI = index_pers_RI(K_index)
  show_pers(K_index, dgms_index, 0)

  ## First, verify all the base case queries work
  for p in [0, 1]:
    for i,j in zip(dgms_index[p]['birth'], dgms_index[p]['death']):
      if j != np.inf:
        assert j-1 >= i, "invalid coordinates"
        assert RI.query(p, i-1, i, j-1, j) == 1

  ## Test that each corner box basecase recovers the pairs
  for p in [0, 1]:
    for i,j in zip(dgms_index[p]['birth'], dgms_index[p]['death']):
      if j != np.inf:
        assert j-1 >= i, "invalid coordinates"
        print(f"Box to query: [{i-1}, {i}] x [{j-1}, {j}]")
        pair = RI.query_pairs(p, i-1, i, j-1, j, verbose=verbose)
        assert len(pair) == 1, "Picked up too many persistence pairs!"
        assert pair['birth'].item() == i, "birth index incorrect!"
        assert pair['death'].item() == j, "death index incorrect!"

  ## Now, swap to a linear extension and verify functionality! 
  IW, wrd = RI._index_weights()
  new_weights = np.sort(np.random.uniform(size=len(wrd), low=0, high=1))
  new_weights = new_weights[wrd['weight'].astype(int)]
  RI._weights[0] = new_weights[wrd['dim'] == 0]
  RI._weights[1] = new_weights[wrd['dim'] == 1]
  RI._weights[2] = new_weights[wrd['dim'] == 2]
  delta = np.sqrt(np.finfo(np.float64).eps)
  for p in [0, 1]:
    for i,j in zip(dgms_index[p]['birth'], dgms_index[p]['death']):
      if j != np.inf:
        assert j-1 >= i, "invalid coordinates"
        a = new_weights[wrd['weight'] == i].item()
        b = new_weights[wrd['weight'] == j].item()
        pairs = RI.query_pairs(p, a-delta, a+delta, b-delta, b+delta, verbose=verbose)
        assert len(pairs) == 1
        assert np.isclose(pairs[0]['birth'], a) and np.isclose(pairs[0]['death'], b)

## Test we can recover the entire diagram exactly, for both index and real-valued persistence
def test_recover_diagram_index():
  np.random.seed(1234)
  verbose = False
  N = 16
  theta = np.linspace(0, 2*np.pi, N, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)
  S = sx.rips_complex(X, p=2)
  K = sx.RankFiltration(S, f=sx.flag_filter(pdist(X)))
  K.order = 'reverse colex'
  K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
  dgms_index = ph(K_index, simplex_pairs=True)
  RI = index_pers_RI(K_index)
  all_boxes = _generate_bfs(0, len(K_index)-1)
  all_pairs = { k : RI.query_pairs(0, i,j,k,l) for ii, (i,j,k,l) in all_boxes.items() }
  all_pairs = [p for p in all_pairs.values() if len(p) > 0]
  all_pairs = np.hstack(all_pairs)
  all_pairs = np.sort(all_pairs, order=['birth', 'death'])
  assert np.allclose(dgms_index[0]['birth'][dgms_index[0]['death'] != np.inf], all_pairs['birth'])
  assert np.allclose(dgms_index[0]['death'][dgms_index[0]['death'] != np.inf], all_pairs['death'])

def test_rips_increasing():
  np.random.seed(1234)
  N = 8
  theta = np.linspace(0, 2*np.pi, N, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)
  K, dgms = rips_example(X, index=False)
  dX = pdist(X)
  H1_b, H1_d = dgms[1]['birth'], dgms[1]['death'] # 15, 41

  RI = SpectralRI(n=len(X), max_dim=2)
  RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
  RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
  RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
  RI._D[0] = RI.boundary_matrix(0)
  RI._D[1] = RI.boundary_matrix(1)
  RI._D[2] = RI.boundary_matrix(2)
  
  # show_pers(K, dgms, p=1, index=False)
  # print(dgms[1])
  # rank_to_comb(5, k=2, order='colex', n=len(X)) # rank 5 <=> edge (2,3) which matches simplex pairs

  # np.sum(np.logical_and(RI._weights[1] >= 0.768, RI._weights[1] <= 0.76805))
  # np.flatnonzero(np.logical_and(RI._weights[1] >= 0.768, RI._weights[1] <= 0.76805))
  # RI._simplices[1][22] # 5 is the creator edge

  # IW, wrd = RI._index_weights()
  # assert IW[1][22] == 15

  # RI.query_pairs(1, 14, 15, 40, 41, method="cholesky", verbose=True, simplex_pairs=True)
  print(dgms[1])
  assert RI.query(1, 1.0, 1.2, 1.8, 2.0, method="cholesky") == 0
  assert RI.query(1, 0.6, 0.8, 1.8, 2.0, method="cholesky") == 1
  assert RI.query(1, 0.6, 0.8, 1.6, 1.8, method="cholesky") == 0
  pairs1 = RI.query_pairs(1, 1.0, 1.2, 1.8, 2.0, method="cholesky", verbose=True, simplex_pairs=True)
  pairs2 = RI.query_pairs(1, 0.6, 0.8, 1.8, 2.0, method="cholesky", verbose=True, simplex_pairs=True)
  pairs3 = RI.query_pairs(1, 0.6, 0.8, 1.6, 1.8, method="cholesky", verbose=True, simplex_pairs=True)
  assert len(pairs1) == 0
  assert len(pairs2) == 1
  assert len(pairs3) == 0


  assert np.allclose(dgms_test['birth'] - H1_b, 0)
  # assert np.allclose(dgms_test['death'] - H1_d, 0)


# def test_something():

#   a,b,c,d = 9, 10, 10, 11
#   delta = 1e-12
#   RI.lower_left(b+delta, c, 1).D.todense()
#   RI.lower_left(a+delta, c, 1).D.todense()
#   RI.lower_left(b+delta, d, 1).D.todense()
#   RI.lower_left(a+delta, d, 1).D.todense()

#   t1_test = np.linalg.matrix_rank(RI.lower_left(i+0, j+0, p=p).D.todense())
#   t2_test = np.linalg.matrix_rank(RI.lower_left(i+1, j+0, p=p).D.todense())
#   t3_test = np.linalg.matrix_rank(RI.lower_left(i+1, j-1, p=p).D.todense())
#   t4_test = np.linalg.matrix_rank(RI.lower_left(i+0, j-1, p=p).D.todense())
  
#   [(b+delta,c), (a+delta,c), (b+delta,d), (a+delta,d)] 
#   RI.query(1, 9, 10, 10, 11, method="direct", summands=True)





# def random_test():
#   ## Persistent Betti 
#   # pattern = [(1,1),(1,0),(0,1),(0,0)]
#   # for x,y in pattern:

#   N = len(K)
#   boxes = _generate_bfs(0, N-1)
#   for i1,i2,j1,j2 in boxes.values():

#     ## Theorem 2 - doesn't quite work right
#     # rows = [slice(i1,j2+1), slice(i1,j1), slice(i1+1,j2+1), slice(i2+1,j1)]
#     # cols = [slice(i1,j2+1), slice(i1,j1), slice(i1+1,j2+1), slice(i2+1,j1)]
#     # ranks = [np.linalg.matrix_rank(D[r,:][:,c]) if np.prod(D[r][:,c].shape) > 0 else 0 for r,c in zip(rows, cols)]

#     ## Equation 3 from Chen & Kerber
#     rows = [slice(i1,N), slice(i1,N), slice(i2+1,N), slice(i2+1,N)]
#     cols = [slice(0,j2+1), slice(0,j1), slice(0,j2+1), slice(0,j1)]
#     sum_pivots = [np.sum(P[r,:][:,c]) for r,c in zip(rows, cols)]
#     rank_R = [np.linalg.matrix_rank(R[r,:][:,c]) if np.prod(R[r][:,c].shape) > 0 else 0 for r,c in zip(rows, cols)]
#     rank_D = [np.linalg.matrix_rank(D[r,:][:,c]) if np.prod(D[r][:,c].shape) > 0 else 0 for r,c in zip(rows, cols)]

#     assert sum_pivots == rank_R, "Number of pivots doesn't match rank"
#     # card_D = sum(s*t for s,t in zip([1,-1,-1,1], rank_D))
#     # card_R = sum(s*t for s,t in zip([1,-1,-1,1], rank_R))
#     # assert card_D == card_R # apparently this isn't true
    
#   ## Verify equation (2.4) in the paper
#   rank = lambda X: np.linalg.matrix_rank(X) if np.prod(X.shape) > 0 else 0
#   for j,i in enumerate(low_entry(R)):
#     if i != -1:
#       t1, t2, t3, t4 = rank(D[i:,:(j+1)]), rank(D[(i+1):,:(j+1)]), rank(D[(i+1):,:j]), rank(D[i:,:j])
#       assert R[i,j] == 1 and (t1 - t2 + t3 - t4) == 1, "Inclusion/exclusion doesn't work out"

#   from combin import comb_to_rank

#   # weights = np.hstack([RI._weights[0], RI._weights[1], RI._weights[2]])
#   # ranks = np.hstack([RI._simplices[0], RI._simplices[1], RI._simplices[2]])
#   # dims = np.hstack([np.repeat(0, len(RI._simplices[0])), np.repeat(1, len(RI._simplices[1])), np.repeat(2, len(RI._simplices[2]))])
#   # wrd = np.array(list(zip(weights, ranks, dims)), dtype=[('weight', 'f4'), ('rank', 'i4'), ('dim', 'i4')])
#   # wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank')))
#   # weights_sorted = weights[np.argsort(wrd_ranking)]
#   # index_map = dict(zip(wrd_ranking, wrd['weight']))
#   # RI._weights[1] = wrd_ranking[wrd['dim'] == 1]
#   # RI._weights[2] = wrd_ranking[wrd['dim'] == 2]
#   def query_oracle(i: int, j: int, k: int, l: int) -> int:
#     return RI.query(1,i,j,k,l)

#   ## Verify query oracle
#   rank = lambda X: np.linalg.matrix_rank(X) if np.prod(X.shape) > 0 else 0
#   for j,i in enumerate(low_entry(R)):
#     if i != -1:
#       t1, t2, t3, t4 = rank(D[i:,:(j+1)]), rank(D[(i+1):,:(j+1)]), rank(D[(i+1):,:j]), rank(D[i:,:j])
#       assert R[i,j] == 1 and (t1 - t2 - t3 + t4) == 1
#       # query_oracle(i, i+1, j, j+1)

  
#   ## Verifies (2.4) again 
#   corner_points = lambda i,j: [(i,j), (i+1,j), (i+1,j-1), (i,j-1)]
#   assert RI.query(0,0,1,4,5) == 0 # empty cell 
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in corner_points(1,6)] == [1,0,0,0]
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in corner_points(2,7)] == [1,0,0,0]
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in corner_points(3,9)] == [1,0,0,0]
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in corner_points(4,10)] == [1,0,0,0]
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in corner_points(5,12)] == [1,0,0,0]

#   ## Verifies persistent Betti number: should all be 2 
#   # pb = []
#   # for j,i in enumerate(low_entry(R[:,:(sx.card(K,0) + sx.card(K,1))])):
#   #   if i != -1:
#   #     qr = RI.query(0,i+1,j-1)
#   #     print(f"({i},{j}) |-> {qr}") # lower-left corner point
#   #     pb.append(qr)
#   # assert np.all(np.array(pb) == 2), "Persistent Betti formulation not correct"

#   ## Verify cancelletions
#   RI.query(0,1,6,summands=True), RI.query(0,0,6,summands=True), RI.query(0,1,7,summands=True), RI.query(0,0,7,summands=True)
  
#   ## Verifies (2.7)  
#   for j,i in enumerate(low_entry(R)):
#     if i != -1:
#       qr = RI.query(0,i-1,i,j,j+1,delta=0)
#       print(f"({i},{j}) |-> {qr}") # lower-left corner point

#   d = 1e-15
#   rect_points = lambda i,j,k,l: [(j+d,k), (i+d,k), (j+d,l), (i+d,l)]
#   [RI.rank(p=1, a=a, b=b, method="direct") for a,b in rect_points(0,1,6,7)]
#   [RI.rank(p=1, a=a, b=b, method="direct") for a,b in rect_points(1,2,7,8)]
#   [RI.rank(p=1, a=a, b=b, method="direct") for a,b in rect_points(2,3,9,10)]
#   [RI.rank(p=1, a=a, b=b, method="direct") for a,b in rect_points(3,4,10,11)]
#   assert [RI.rank(p=1, a=a, b=b, method="direct") for a,b in rect_points(4,5,12,13)] == [0,0,0,1]
#   RI.query(0,4,5,12,13, summands=True)

#   RI.query(0,1,6) == 2
#   assert RI.query(0, 2, 11) == 1
#   assert RI.query(0, 0, 0) == 1
#   assert RI.query(0, -1e-15, -1e-15) == 0
#   assert RI.query(0, 2, 5) == 3

# # p = len(K[j][1]) - 1
# # RI.query(p-1, i, i+1, j, j+1)

#   # query_oracle()
#   # pass


#   ## Recover (3,9) but shouldn't?
#   show_pers(K, dgms_index, 0)
#   # RI.query_pairs(0, 3, 6, 9, 12, verbose=True)
#   assert RI.query(0,3,6,9,12) == 2
#   assert RI.query(0,3,7,9,12) == 2
#   assert RI.query(0,3,7,9,13) == 2
#   points_in_box(3,6,9,12, query=query_oracle(0))
#   RI.query_pairs(0,3,6,9,12, verbose=True)
    
#     ## Recovers correct high persistence pair, but base casee needs to be amended
#   ## right now the base case is like i, i+1, i+2, i+2
#   ## Should be i, i+1, i+1, i+2
#   ## Fixed! 
#   # query_oracle(680,681,682,682)

# def test_rank_query():
#   np.random.seed(1234)
#   N = 8
#   theta = np.linspace(0, 2*np.pi, N, endpoint=False)
#   X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)
#   dX = pdist(X)

#   ## Start with a reverse colexicographically filtered complex with index filter values
#   K = sx.RankFiltration(sx.rips_filtration(X, np.inf, p=2))
#   K.order = 'reverse colex'
#   K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
#   dgms_index = ph(K_index, simplex_pairs=True)
#   H1_b, H1_d = dgms_index[1]['birth'], dgms_index[1]['death']

#   def query_oracle_truth(i: int, j: int, k: int, l: int) -> int:
#     in_birth = np.logical_and(i <= H1_b, H1_b <= j)
#     in_death = np.logical_and(k <= H1_d, H1_d <= l)
#     return np.sum(np.logical_and(in_birth, in_death))

#   ## Generate all persistence pairs in the diagram
#   from spirit.query import _generate_bfs
#   all_boxes = _generate_bfs(0, len(K)-1)
#   all_pairs = { k : points_in_box(*box, query_oracle) for k, box in all_boxes.items() }
#   all_pairs = [np.ravel([(b,d) for b,d in p.items()]) for p in all_pairs.values() if len(p) > 0]
#   all_pairs = np.hstack(all_pairs).reshape((len(list(collapse(all_pairs))) // 2, 2))
#   all_pairs = all_pairs[np.lexsort(np.rot90(all_pairs)),:]
#   assert np.allclose(all_pairs - np.c_[dgms_index[1]['birth'], dgms_index[1]['death']], 0)


#   from pbsig.vis import figure_dgm
#   show(figure_dgm(dgms_index[1]))

#   ## Test query model 
#   RI = SpectralRI(n=N, max_dim=2)
#   RI.construct(dX, p=0, apparent=True, discard=False, filter="flag")
#   RI.construct(dX, p=1, apparent=True, discard=False, filter="flag")
#   RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")
#   RI._D[0] = RI.boundary_matrix(0)
#   RI._D[1] = RI.boundary_matrix(1)
#   RI._D[2] = RI.boundary_matrix(2)
#   # assert len(RI._weight[p]) == len(RI._weight[p]) and len(RI._weight[p]) 
#   weights = np.hstack([RI._weights[0], RI._weights[1], RI._weights[2]])
#   ranks = np.hstack([RI._simplices[0], RI._simplices[1], RI._simplices[2]])
#   dims = np.hstack([np.repeat(0, len(RI._simplices[0])), np.repeat(1, len(RI._simplices[1])), np.repeat(2, len(RI._simplices[2]))])
#   wrd = np.array(list(zip(weights, ranks, dims)), dtype=[('weight', 'f4'), ('rank', 'i4'), ('dim', 'i4')])
#   wrd_ranking = np.argsort(np.argsort(wrd, order=('weight', 'dim', 'rank')))
#   weights_sorted = weights[np.argsort(wrd_ranking)]
#   index_map = dict(zip(wrd_ranking, wrd['weight']))
#   # inv_index_map = lambda x: np.searchsorted(weights_sorted, x)
#   # def query_oracle(i: int, j: int, k: int, l: int) -> int:
#   #   return RI.query(1, index_map[i], index_map[j], index_map[k], index_map[l], method="cholesky")

#   ## Ahh, don't use an index_map, just change the weights!
#   RI._weights[1] = wrd_ranking[wrd['dim'] == 1]
#   RI._weights[2] = wrd_ranking[wrd['dim'] == 2]
#   def query_oracle(i: int, j: int, k: int, l: int) -> int:
#     return RI.query(1,i,j,k,l)

#   all_boxes = _generate_bfs(0, max(index_map.keys()))
#   all_pairs_test = { k : points_in_box(*box, query_oracle) for k, box in all_boxes.items() }
#   points_in_box(*all_boxes[0], query_oracle)

#   query_oracle(0,28,29,57)       # this is exclusive on [28,29]
#   query_oracle_truth(0,28,29,57) # this is inclusive on [28,29]

#   query_oracle(0,14,29,57)
#   query_oracle(15,28,29,57)

#   query_oracle_truth(0,14,29,57)
#   query_oracle_truth(15,28,29,57)

#   query_oracle(14,28,115,229)

#   points_in_box(0, 114, 115, 229, query=query_oracle)

#   i,j,k,l = 0, 114, 115, 229
#   positive = {}
#   bisection_tree(i, j, k, l, mu_init, query_oracle, positive, verbose=True)
#   query_oracle(0,28,115,229)

#   # pairs = { c : find_negative(c, j1, j2, query) for c, (j1, j2) in positive.items() }

#   ## Generate all persistent pairs with persistence greater than threshold 
#   ## TODO: 
