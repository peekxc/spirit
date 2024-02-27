import numpy as np 
import splex as sx 
from spirit.query import bisection_tree, points_in_box, find_negative
from scipy.spatial.distance import pdist
from ripser import ripser

def test_bisection():
  np.random.seed(1234)
  N = 16
  theta = np.linspace(0, 2*np.pi, N, endpoint=False)
  X = np.c_[np.cos(theta), np.sin(theta)] + np.random.uniform(size=(N,2), low=0, high=0.01)
  H1_pt = ripser(X)['dgms'][1][0]

  ## Start with a reverse colexicographically filtered complex with index filter values
  K = sx.rips_filtration(X, np.inf, p=2)
  K = sx.RankFiltration(K)
  K.order = 'reverse colex'
  # K.reindex(np.arange(len(K)))

  diam_f = sx.flag_filter(pdist(X))
  from pbsig.persistence import ph 
  dgms = ph(K, simplex_pairs=True)

  K_index = sx.RankFiltration({i:s for i,(d,s) in enumerate(K)}.items())
  dgms_index = ph(K_index, simplex_pairs=True)
  H1_b, H1_d = dgms_index[1]['birth'], dgms_index[1]['death']

  def query_oracle(i: int, j: int, k: int, l: int) -> int:
    # in_birth = K[i][0] <= H1_pt[0] and H1_pt[0] <= K[j][0]
    # in_death = K[k][0] <= H1_pt[1] and H1_pt[1] <= K[l][0]
    in_birth = np.logical_and(i <= H1_b, H1_b <= j)
    in_death = np.logical_and(k <= H1_d, H1_d <= l)
    return np.sum(np.logical_and(in_birth, in_death))

wut = points_in_box(0, 120+16, 120+16+1, 16+120+560-1, query_oracle)
dgms_index[1]['death']-dgms_index[1]['birth']

## Generate all persistence pairs in the diagram
from spirit.query import _generate_bfs
all_boxes = _generate_bfs(0, len(K)-1)
all_pairs = { k : points_in_box(*box, query_oracle) for k, box in all_boxes.items() }
all_pairs = np.array([np.ravel(tuple(p.items())) for p in all_pairs.values() if len(p) > 0])
all_pairs = all_pairs[np.lexsort(np.rot90(all_pairs)),:]
assert np.allclose(all_pairs - np.c_[dgms_index[1]['birth'], dgms_index[1]['death']], 0)


## Generate all persistent pairs with persistence greater than threshold 
## TODO: 


from pbsig.vis import figure_dgm, show
show(figure_dgm(dgms_index[1]))


  # d = {31: (137, 695), 136: (137, 695)}
  # { c : find_negative(c, j1, j2, query_oracle) for c, (j1, j2) in d.items() }
  # c, (j1,j2) = 31, (137, 695)
  # find_negative(c, j1, j2, query_oracle)


  query_oracle(31,32,137,695)
  query_oracle(31,32,137,695)

  # ## Complete complex 
  # st = sx.simplicial_complex([[v] for v in np.arange(len(X))], form='tree')
  # st.insert(combinations(st.vertices, 2))
  # st.expand(k=2)