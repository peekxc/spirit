# %% 
from typing import *
import scipy as sp
import numpy as np 
import splex as sx
from scipy.spatial.distance import pdist 
from scipy.sparse.linalg import eigsh
from spirit.apparent_pairs import SpectralRI
from pbsig.persistence import sliding_window
from combin import rank_to_comb, comb_to_rank
from ripser import ripser
from math import comb
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.io import output_notebook
output_notebook()

# %% 
f = lambda t: np.real(np.exp(1j*t) + np.exp(1j*(2**(1/2))*t) + np.exp(1j*(3**(1/2))*t))
SW = sliding_window(f, bounds=(0, 400))

# %% Plot periodic function
from pbsig.persistence import sw_parameters
M = 3
_, tau_opt = sw_parameters(bounds=(0, 400), d=M, tau=49.325)
dom = np.linspace(0, 400, 1200)

# p = figure(width=850, height=150)
# p.line(dom, f(dom), line_width=1.5)
# p.rect(x=w/2, y=0, width=w, height=4, fill_alpha=0.20, fill_color='gray', line_alpha=0, line_width=0)
# p.toolbar_location = None
# show(p)

from landmark import landmarks
X_delay = SW(2000,d=M,tau=tau_opt)
ind = landmarks(X_delay,800)
dgms = ripser(X_delay[ind], maxdim=2)['dgms']

from pbsig.vis import figure_dgm
p = figure_dgm(dgms[1])
q = figure_dgm(dgms[2])
show(row(p, q))
