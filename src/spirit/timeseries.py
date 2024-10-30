from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def sw_parameters(bounds: tuple, d: int = None, tau: float = None, w: float = None, L: int = None):
	"""Computes slidding window parameters.

	Given an interval [a,b] and any two slidding window parameters (d, w, tau, L),
	returns the necessary parameters (d, tau) needed to construct Takens slidding window embedding.
	"""
	if not (d is None) and not (tau is None):
		return (d, tau)
	elif not (d is None) and not (w is None):
		return (d, w / d)
	elif not (d is None) and not (L is None):
		w = bounds[1] * d / (L * (d + 1))
		return (d, w / d)
	elif not (tau is None) and not (w is None):
		d = int(w / tau)
		return (d, tau)
	elif not (tau is None) and not (L is None):
		d = bounds[1] / (L * tau) - 1
		return (d, tau)
	elif not (w is None) and not (L is None):
		d = np.round(1 / ((bounds[1] - w * L) / (w * L)))
		assert d > 0, "window*L must be less than the entire time duration!"
		return (d, w / d)
	else:
		raise ValueError("Invalid combination of parameters given.")


def sliding_window(f: Union[ArrayLike, Callable], bounds: tuple = (0, 1)):
	"""Sliding Window Embedding of a time series.

	Returns a function which generates a n-point slidding window point cloud of a fixed time-series/function _f_.

	The returned function has the parameters
	  - n: int = number of point to generate the embedding
	  - d: int = dimension-1 of the resulting embedding
	  - w: float = (optional) window size each (d+1)-dimensional delay coordinate is derived from
	  - tau: float = (optional) step size
	  - L: int = (optional) expected number of periods, if known
	The parameter n and d must be supplied, along with exactly one of 'w', 'tau' or 'L'.
	"""
	assert isinstance(f, Callable) or isinstance(f, np.ndarray), "Time series must be function or array like"

	# Assume function like, defined on [0, 1]
	# if isinstance(f, Callable):  # Construct a continuous interpolation via e.g. cubic spline
	def sw(n: int, d: int = None, tau: float = None, w: float = None, L: int = None):
		"""Creates a slidding window point cloud over 'n' windows"""
		d, tau = sw_parameters(bounds, d=d, tau=tau, w=w, L=L)
		T = np.linspace(bounds[0], bounds[1] - d * tau, n)
		delay_coord = lambda t: np.array([f(t + di * tau) for di in range(d + 1)])
		X = np.array([delay_coord(t) for t in T])
		return X

	return sw
	# else:
	# 	d, tau = sw_parameters(bounds, d=d, tau=tau, w=w, L=L)
	# 	T = np.linspace(bounds[0], bounds[1] - d * tau, n).astype(int)
	# 	delay_coord = lambda t: np.array([f[int(t + di * tau)] for di in range(d + 1)])
	# return sw
