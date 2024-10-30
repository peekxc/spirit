from functools import partial
import itertools as it
from typing import Callable, Union
from functools import partial

from array import array
import eagerpy as ep
import numpy as np
import splex as sx
import torch
from bokeh.plotting import figure, show
from numpy.typing import ArrayLike
from bokeh.palettes import Sunset
from tqdm import tqdm


from .plotting import BlockMapReduce, step_dn, step_up


class RankInvariantHolder:
	def __init__(self, params, EIG_vals: list) -> None:
		self.params = params  # reference to true parameters
		self.EIG_vals = EIG_vals
		# self.params = params.detach().numpy()

	def __call__(self, g: Callable[[torch.tensor], torch.tensor] = None, **kwargs) -> torch.tensor:
		if g is None:
			EIG_sums = [torch.sum(s) for s in self.EIG_vals]
		else:
			EIG_sums = [torch.sum(g(s, **kwargs)) for s in self.EIG_vals]
		pb_num = (EIG_sums[0] + EIG_sums[3]) - (EIG_sums[1] + EIG_sums[2])
		return pb_num

	def __array__(self, dtype=None):
		return self.__call__().detach().numpy().astype(dtype=dtype)

	def __float__(self) -> float:
		return self.__array__().item()

	def __repr__(self) -> str:
		return "RI: " + str(self.__call__())

	def backward(self, g: Callable[[torch.tensor], torch.tensor] = None, **kwargs) -> torch.tensor:
		pb_num = self.__call__(g, **kwargs)
		pb_num.backward(retain_graph=True)
		return self.params.grad


class RipsRankInvariant:
	def __init__(self, X: np.ndarray) -> None:
		assert X.ndim == 2, "Must be a two dimensional ndarray."
		self.P = torch.tensor(X, requires_grad=True)
		self.n = X.shape[0]
		S = sx.rips_complex(X, p=2, radius=np.inf)
		DM_shape = [sx.card(S, 0)] * 2
		D1 = sx.boundary_matrix(S, 1)
		D2 = sx.boundary_matrix(S, 2)
		self.D1_t = torch.tensor(D1.todense(), dtype=torch.float64)
		self.D2_t = torch.tensor(D2.todense(), dtype=torch.float64)
		self.EI = np.ravel_multi_index(S.edges.T, dims=DM_shape, order="C")
		self.TI = np.ravel_multi_index(S.triangles[:, [0, 1]].T, dims=DM_shape, order="C")
		self.TJ = np.ravel_multi_index(S.triangles[:, [0, 2]].T, dims=DM_shape, order="C")
		self.TK = np.ravel_multi_index(S.triangles[:, [1, 2]].T, dims=DM_shape, order="C")

	@staticmethod
	def regularize(x: ArrayLike, eps: float = 0.80):
		x = ep.astensor(x)
		y = 1.0 - (-x / eps).exp()
		return y.raw
		# return 1.0 - torch.exp(-x / eps)

	## https://stackoverflow.com/questions/13029254/python-property-callable
	def transform(self, p: torch.tensor):
		Pt = self._transform(self.P, p)
		assert Pt.shape == self.P.shape, "Must return the same shape point cloud"
		assert hasattr(Pt, "grad_fn"), "Must be a differentiable function tracked with autograd"
		# P_diff = torch.abs((Pt - Pt[:, np.newaxis]))
		# ## NOTE: using sqrt here causes nan's
		# DM = torch.maximum(torch.pow(P_diff, 2).sum(axis=2), torch.tensor([0.0]))
		# DV = DM.ravel()
		DM = torch.float_power(torch.cdist(Pt, Pt), 2)
		DV = DM.ravel()
		self.v_diam = torch.zeros(self.n, dtype=torch.float64)
		self.e_diam = DV[self.EI]
		self.t_diam = torch.maximum(torch.maximum(DV[self.TI], DV[self.TJ]), DV[self.TK])
		return Pt

	@property
	def transform_fn(self):
		return self._transform

	@transform_fn.setter
	def transform_fn(self, value: Callable[[torch.tensor, torch.tensor], torch.tensor]):
		assert isinstance(value, Callable), "Transform function must be a callable"
		self._transform = value

	@staticmethod
	def laplacian(D: torch.tensor, f: torch.tensor, q: torch.tensor):
		return torch.diag(f) @ D @ torch.diag(q) @ D.T @ torch.diag(f)

	## NOTE: It would be ideal to have P + diameters sharing memory
	## Maybe could return a torch.Function here that stores the eigenvalues
	## This is sort of not ideal but of regularize
	def forward(self, p: torch.tensor, a: float, b: float, w: float = 0.0):
		assert isinstance(p, torch.Tensor), "For now the input to forward should be a torch tensor"
		self.transform(p)
		assert hasattr(self, "t_diam"), "Must call .transform(params, fun) first."
		a_, b_, w_ = a**2, b**2, w**2  ## this is to match the suqared diameter function above
		v_wght_a_exc, v_wght_a_inc = step_up(self.v_diam, a=a_, w=w_), step_dn(self.v_diam, a=a_, w=w_)
		e_wght_a_exc, e_wght_a_inc = step_up(self.e_diam, a=a_, w=w_), step_dn(self.e_diam, a=a_, w=w_)
		e_wght_b_exc, e_wght_b_inc = step_up(self.e_diam, a=b_, w=w_), step_dn(self.e_diam, a=b_, w=w_)
		t_wght_b_exc, t_wght_b_inc = step_up(self.t_diam, a=b_, w=w_), step_dn(self.t_diam, a=b_, w=w_)
		self.EIG_vals = [None] * 4
		self.EIG_vals[0] = e_wght_a_inc
		self.EIG_vals[1] = torch.linalg.eigvalsh(self.laplacian(self.D1_t, v_wght_a_inc, e_wght_a_inc))
		self.EIG_vals[2] = torch.linalg.eigvalsh(self.laplacian(self.D2_t, e_wght_b_inc, t_wght_b_inc))
		self.EIG_vals[3] = torch.linalg.eigvalsh(self.laplacian(self.D2_t, e_wght_a_exc, t_wght_b_inc))
		return RankInvariantHolder(p, self.EIG_vals)

	def __call__(
		self, p: Union[torch.tensor, np.ndarray] = None, a: float = 0.0, b: float = 0.0, w: float = 0.0, grad: bool = False
	):
		args = (a, b, w)

		def _objective(p: Union[torch.tensor, np.ndarray], eps: float = 1.0):
			is_ndarray = isinstance(p, np.ndarray)
			p = torch.tensor(p, requires_grad=True, dtype=torch.float64) if is_ndarray else p
			assert isinstance(p, torch.Tensor), "forward only accepts torch tensors"
			betti_num = self.forward(p, *args)
			betti_num.backward(self.regularize, eps=eps)
			obj = np.asarray(betti_num) if is_ndarray else betti_num
			jac = None
			if grad:
				jac = p.grad.detach().numpy() if is_ndarray else p.grad
			return (obj, jac) if grad else obj

		return _objective if p is None else _objective(p)

	# def backward(self, g: Callable[[torch.tensor], torch.tensor]) -> torch.tensor:
	# 	EIG_sums = [torch.sum(g(s)) for s in self.EIG_vals]
	# 	pb_num = (EIG_sums[0] + EIG_sums[3]) - (EIG_sums[1] + EIG_sums[2])
	# 	pb_num.backward(retain_graph=True)
	# 	return self.params.grad

	def gradient(self, p: Union[torch.tensor, np.ndarray], a: float, b: float, w: float = 0.0) -> torch.tensor:
		return self.__call__(p, a, b, w, grad=True)[1]

	def precompute_contour(self, a: float, b: float, w: float, xr: tuple = (0, 1), yr: tuple = (0, 1), res: int = 50):
		fixed_params = (a, b, w)
		gx, gy = np.meshgrid(np.linspace(*xr, res), np.linspace(*yr, res))
		# term1, term2, term3, term4 = array("d"), array("d"), array("d"), array("d")
		term1, term2, term3, term4 = [], [], [], []
		for ii, (p1, p2) in tqdm(enumerate(zip(gx.ravel(), gy.ravel())), total=res**2):
			p = torch.tensor(np.array([p1, p2]), dtype=torch.float64, requires_grad=True)
			self.forward(p, *fixed_params)  # populates EIG_vals
			ew = [v.detach().numpy() for v in self.EIG_vals]
			# info[ii] = {"ri": ew, "xy": (p1, p2)}
			term1.append(ew[0])
			term2.append(ew[1])
			term3.append(ew[2])
			term4.append(ew[3])

		## Collect the terms
		self.T1 = BlockMapReduce(term1, False)
		self.T2 = BlockMapReduce(term2, False)
		self.T3 = BlockMapReduce(term3, False)
		self.T4 = BlockMapReduce(term4, False)
		self.gx = gx
		self.gy = gy

	def contour_2D(self, eps: float = 1.0, n_levels: int = 8):
		assert hasattr(self, "gx") and hasattr(self, "T1")
		g = partial(self.regularize, eps=eps)
		Z = (self.T1(g) - self.T2(g) - self.T3(g) + self.T4(g)).reshape(self.gx.shape)
		p = figure(width=575, height=500, match_aspect=True)
		delta = np.ptp(Z.ravel()) * 0.05
		levels = np.linspace(np.min(Z) - delta, np.max(Z) + delta, n_levels)
		contour_renderer = p.contour(self.gx, self.gy, Z, levels, fill_color=Sunset, line_color="black")
		colorbar = contour_renderer.construct_color_bar()
		p.add_layout(colorbar, "right")
		return p
