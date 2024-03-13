import numpy as np
from spirit.apparent_pairs import SpectralRI
from scipy.spatial.distance import squareform
from pbsig.persistence import sliding_window
from scipy.spatial.distance import pdist, squareform

f = lambda t: np.cos(t) + np.cos(3*t)
SW = sliding_window(f, bounds=(0, 12*np.pi))
tau_opt = 0.299199300341885

N, M = 50, 2
X = SW(n=N, d=M, tau=tau_opt)
dX = pdist(X)
DX = squareform(dX)
RI = SpectralRI(n=N, max_dim=2)
RI.construct(dX, p=0, apparent=False, discard=False, filter="flag")
RI.construct(dX, p=1, apparent=False, discard=False, filter="flag")

for i in range(20):
  RI.construct(dX, p=2, apparent=True, discard=True, filter="flag")