import itertools as it
import os
import subprocess
import tempfile
from itertools import *
from typing import Any, Callable, Collection, Sequence, Union

import numpy as np
from bokeh.models import Span
from bokeh.plotting import figure, show
from numpy.typing import ArrayLike


## Meta-programming to the rescue!
def valid_parameters(el: Any, prefix: str = "", exclude: list = [], **kwargs):
	"""Extracts valid parameters for a bokeh plotting element.

	This function takes as input a bokeh model (i.e. figure, Scatter, MultiLine, Patch, etc.) and a set of keyword arguments prefixed by 'prefix',
	and extracts from the the given keywords a dictionary of valid parameters for the given element, with the chosen prefix removed.

	Example:

	  >>> valid_parameters(MultiLine, prefix="multiline_", multiline_linewidth=2.0)
	  # { 'linewidth' : 2.0 }

	Parameters:
	  el: bokeh model or glyph element with a .parameters() function
	  prefix: prefix to extract. Defaults to empty string.
	  kwargs: keyword arguments to extract the valid parameters from.

	"""
	assert hasattr(el, "parameters"), f"Invalid bokeh element '{type(el)}'; must have valid parameters() call"
	param_names = {param[0].name for param in el.parameters()}
	stripped_params = {k[len(prefix) :] for k in kwargs.keys() if k.startswith(prefix)}
	valid_params = {p: kwargs[prefix + p] for p in (stripped_params & param_names) if not p in exclude}
	return valid_params


def as_dgm(dgm: np.ndarray) -> np.ndarray:
	if (
		isinstance(dgm, np.ndarray)
		and dgm.dtype.names is not None
		and "birth" in dgm.dtype.names
		and "death" in dgm.dtype.names
	):
		return dgm
	dgm = np.atleast_2d(dgm)
	dgm_dtype = [("birth", "f4"), ("death", "f4")]
	return np.array([tuple(pair) for pair in dgm], dtype=dgm_dtype)


def smoothstep(x: np.array, a: float, w: float):
	"""Computes a smooth approximation of the step function centered as `a` with width `w`."""
	y = np.clip(x, a, a + w)
	z = (y - np.array([a])) / (np.array([w]))
	return 3.0 * np.power(z, 2) - 2.0 * np.power(z, 3)


def step_up(x: ArrayLike, a: float, w: float) -> ArrayLike:
	"""Computes a smooth approximation of the step (up) function centered as `a` with width `w`.

	Produces a smoother version of the step function S centered at 'a', with properties:
		1. sign+(S(x)) = sign+(x) for all x
		2. S(x) is continuous and differentiable
		3. 0 <= S(x) <= 1, with S(x) = 0 for all x <= a and 1 for all a + w <= x
	"""
	x_typename = type(x).__name__
	if x_typename == "ndarray":
		y = np.clip(x, a, a + w)
		z = (y - np.array([a])) / (np.array([w]))
		return 3.0 * np.power(z, 2) - 2.0 * np.power(z, 3)
	elif x_typename == "Tensor":
		import torch  # noqa: PLC0415

		y = torch.clip(x, a, a + w)
		z = (y - torch.tensor([a])) / (torch.tensor([w]))
		return 3.0 * torch.pow(z, 2) - 2.0 * torch.pow(z, 3)
	else:
		raise ValueError(f"Invalid ArrayLike '{x_typename}' for argument x.")


def step_dn(x: ArrayLike, a: float, w: float) -> ArrayLike:
	"""Computes a smooth approximation of the step (down) function centered as `a` with width `w`.

	Produces a smoother version of the step function S centered at 'a', with properties:
		1. sign+(S(x)) = sign+(x) for all x
		2. S(x) is continuous and differentiable
		3. 0 <= S(x) <= 1, with S(x) = 0 for all x <= a and 1 for all a + w <= x
	"""
	x_typename = type(x).__name__
	if x_typename == "ndarray":
		y = np.clip(x, a - w, a)
		z = 1.0 - np.abs((y - np.array([a])) / (np.array([w])))
		z = 3.0 * np.power(z, 2) - 2.0 * np.power(z, 3)
		return 1.0 - z
	elif x_typename == "Tensor":
		import torch  # noqa: PLC0415

		y = torch.clip(x, a - w, a)
		z = torch.tensor([1.0]) - torch.abs((y - torch.tensor([a])) / (torch.tensor([w])))
		z = 3.0 * torch.pow(z, 2) - 2.0 * torch.pow(z, 3)
		return torch.tensor([1.0]) - z
	else:
		raise ValueError(f"Invalid ArrayLike '{x_typename}' for argument x.")


def figure_smoothstep(a: float, w: float, up: bool = True):
	dom = np.linspace(a - 1.0 - w, a + 1.0 + w, 1500)
	p = figure(width=250, height=250, match_aspect=True)
	if up:
		p.line(dom, step_up(dom, a, w))
		p.scatter([a], step_up(np.array([a]), a, w))
		# p.scatter(dom, smoothstep(dom, a, w))
	else:
		p.line(dom, step_dn(dom, a, w))
		p.scatter([a], step_dn(np.array([a]), a, w))
	return p


def figure_dgm(dgm: Sequence[tuple] = None, pt_size: int = 5, show_filter: bool = False, essential_val=None, **kwargs):
	default_figkwargs = dict(width=300, height=300, match_aspect=True, aspect_scale=1, title="Persistence diagram")
	fig_kwargs = default_figkwargs.copy()
	if dgm is None or len(dgm) == 0:
		fig_kwargs["x_range"] = (0, 1)
		fig_kwargs["y_range"] = (0, 1)
		min_val = 0
		max_val = 1
	else:
		dgm = as_dgm(dgm)
		max_val = max(np.ravel(dgm["death"]), key=lambda v: v if v != np.inf else -v)
		max_val = max_val if max_val != np.inf else max(dgm["birth"]) * 5
		min_val = min(np.ravel(dgm["birth"]))
		min_val = min_val if min_val != max_val else 0.0
		delta = abs(min_val - max_val)
		min_val, max_val = min_val - delta * 0.10, max_val + delta * 0.10
		fig_kwargs["x_range"] = (min_val, max_val)
		fig_kwargs["y_range"] = (min_val, max_val)

	## Parameterize the figure
	from bokeh.models import PolyAnnotation

	fig_kwargs = valid_parameters(figure, **(fig_kwargs | kwargs))
	p = kwargs.get("figure", figure(**fig_kwargs))
	p.xaxis.axis_label = "Birth"
	p.yaxis.axis_label = "Death"
	polygon = PolyAnnotation(
		fill_color="gray",
		fill_alpha=1.0,
		xs=[min_val - 100, max_val + 100, max_val + 100],
		ys=[min_val - 100, min_val - 100, max_val + 100],
		line_width=0,
	)
	p.add_layout(polygon)
	# p.patch([min_val-100, max_val+100, max_val+100], [min_val-100, min_val-100, max_val+100], line_width=0, fill_color="gray", fill_alpha=0.80)

	## Plot non-essential points, where applicable
	if dgm is not None and any(dgm["death"] != np.inf):
		x = dgm["birth"][dgm["death"] != np.inf]
		y = dgm["death"][dgm["death"] != np.inf]
		p.scatter(x, y, size=pt_size)

	## Plot essential points, where applicable
	if dgm is not None and any(dgm["death"] == np.inf):
		x = dgm["birth"][dgm["death"] == np.inf]
		essential_val = max_val - delta * 0.05 if essential_val is None else essential_val
		y = np.repeat(essential_val, sum(dgm["death"] == np.inf))
		s = Span(dimension="width", location=essential_val, line_width=1.0, line_color="gray", line_dash="dotted")
		s.level = "underlay"
		p.add_layout(s)
		p.scatter(x, y, size=pt_size, color="red")

	return p


from bokeh.models import Arrow, ArrowHead, ColumnDataSource, OpenHead
from bokeh.palettes import Sunset


## Speeds up the application of map-reduce type operations over fixed collections of arrays (by about 250x)
class BlockMapReduce:
	def __init__(self, x: list, eliminate_zeros: bool = False):
		if eliminate_zeros:
			x = [np.array([0]) if np.allclose(s, 0.0) else np.array(s)[~np.isclose(s, 0)] for s in x]
		self.arr_len = np.array([len(s) for s in x], dtype=np.int64)
		self.indices = np.append([0], np.cumsum(self.arr_len)[:-1])
		self.data = np.concatenate(x)
		self._out = np.zeros(len(self.arr_len), dtype=self.data.dtype)

	def __call__(self, g: np.ufunc = None) -> np.ndarray:
		temp_data = g(self.data) if g is not None else self.data
		np.add.reduceat(temp_data, self.indices, out=self._out)
		return self._out

	def __repr__(self) -> str:
		return f"Block Mapper w/ blocks: {np.diff(self.indices[:3])}, ..., {np.diff(self.indices[-4:])}"


def animate_seq(
	figures: Union[str, Sequence],
	output_fn: str,
	fps: int = 10,
	scale: float = 1.0,
	format: str = "mp4",
	output_dn: str = None,
	width: int = 250,
	height: int = 250,
) -> str:
	from bokeh.io import export_png, export_svg
	from tqdm import tqdm
	from selenium import webdriver
	from selenium.webdriver.chrome.options import Options

	## Initialize headless browser session
	chrome_options = Options()
	chrome_options.add_argument("--headless")
	driver = webdriver.Chrome(options=chrome_options)

	tmpdir = tempfile.TemporaryDirectory().name if output_dn is None else output_dn
	if not os.path.isdir(tmpdir):
		os.makedirs(tmpdir)

	file_paths = []
	w, h = int(width), int(height)
	print(f"Saving outputs to: {tmpdir}")
	tmp_dir_name = str(tmpdir)
	for i, fig in tqdm(enumerate(figures)):
		file_path = os.path.join(tmp_dir_name, f"frame_{i:05d}.png")
		export_png(fig, filename=file_path, webdriver=driver, scale_factor=scale)
		file_paths.append(file_path)
	driver.quit()

	# Convert PNGs to MP4 using ffmpeg
	output_file = f"{output_fn}.{format}"
	ffmpeg_cmd = ["ffmpeg", "-y"]
	ffmpeg_cmd += ["-i", os.path.join(tmp_dir_name, "frame_%05d.png")]
	ffmpeg_cmd += ["-vf", f"scale={w}:{h}"]
	ffmpeg_cmd += ["-c:v", "libx264"]
	ffmpeg_cmd += ["-r", str(fps)]
	ffmpeg_cmd += ["-pix_fmt", "yuv420p"]
	ffmpeg_cmd += ["-loop", "0", output_file]

	# subprocess.run(ffmpeg_cmd, check=True)
	cmd = " ".join(ffmpeg_cmd)
	status = subprocess.call(f"source ~/.bash_profile && {cmd}", shell=True)
	return status

	# # fmt: off
	# stream = ffmpeg.input(os.path.join(tmpdir.name, 'frame_%05d.png'), framerate=10) \
	# stream = stream.output(output_fn, loop=0)
	# stream.run()
	# fmt: on


# class ObjectiveSurface2d:
# 	"""Represents a 2d objective surface of a function defined by a vector of inputs at each Z[i,j]:"""

# 	def __init__(self, Callable):
# 		# self.blocks = list(map(BlockMapReduce, blocks))
# 		pass

# 	def init_grid(self, obj_func: Callable, x_rng: tuple = (0, 1), y_rng: tuple = (0, 1), n_points: int = 150):
# 		self.x = np.linspace(*x_rng, num=n_points)
# 		self.y = np.linspace(*y_rng, num=n_points)
# 		self.Z = np.zeros((n_points, n_points), dtype=np.float64)
# 		for i, j in it.product(2 * [range(n_points)]):
# 			self.Z[i, j] = obj_func([self.x[i], self.y[j]])

# 	def contour(self):
# 		fig = figure(width=575, height=500, match_aspect=True)
# 		delta = np.ptp(self.Z.ravel()) * 0.05
# 		levels = np.linspace(np.min(self.Z) - delta, np.max(self.Z) + delta, 15)
# 		X, Y = np.meshgrid(self.x, self.y)
# 		contour_renderer = self.fig.contour(X, Y, self.Z, levels, fill_color=Sunset, line_color="black")
# 		colorbar = contour_renderer.construct_color_bar()
# 		fig.add_layout(colorbar, "right")
# 		return fig

# 	def mapreduce():
# 		pass


# def gradient_field(self, x: np.ndarray, y: np.ndarray, Z: np.ndarray):
# 	delta = np.min(np.diff(Rx))
# 	dx, dy = delta * 0.75, delta * 0.75
# 	xs_arr, xe_arr, ys_arr, ye_arr = [], [], [], []
# 	x_sca, y_sca = [], []
# 	oh = OpenHead(line_color="black", line_width=1.0, size=3)

# 	grads = np.array([V["grad"] for V in info.values()])
# 	points = np.array([V["xy"] for V in info.values()])
# 	grad_mag = np.linalg.norm(grads, axis=1)

# 	unit_vec = grads.copy()
# 	near_zero = np.isclose(grad_mag, 0.0, atol=1e-6)
# 	unit_vec *= np.reciprocal(np.where(near_zero, 1.0, grad_mag))[:, np.newaxis]
# 	xs_arr, ys_arr = points[~near_zero].T
# 	xe_arr, ye_arr = (
# 		points[~near_zero] + ((unit_vec * np.array([dx, dy])) * 0.01 * grad_mag[:, np.newaxis])[~near_zero]
# 	).T

# 	cds = ColumnDataSource(dict(x_start=xs_arr, y_start=ys_arr, x_end=xe_arr, y_end=ye_arr))
# 	p.add_layout(Arrow(end=oh, line_color="black", line_width=1, source=cds))
# 	show(p)
