from typing import Callable, Optional, Union

import numpy as np
import torch
from scipy.optimize import OptimizeResult  # todo
from tqdm import tqdm


def gradient_descent(
	fun: Callable,
	x0: np.ndarray,
	args: tuple = (),
	jac: Union[Callable, bool, None] = None,
	lr: float = 0.01,
	atol: float = 1e-6,
	maximize: bool = False,
	maxiter: int = 1000,
	callback: Optional[Callable] = None,
	verbose: bool = False,
	**kwargs: dict,
):
	steps = []
	params = torch.tensor(x0, dtype=torch.float64, requires_grad=True) if not isinstance(x0, torch.Tensor) else x0
	optimizer = torch.optim.SGD([params], lr=lr, maximize=maximize, **kwargs)
	if jac is None or (isinstance(jac, bool) and jac):
		call_obj_grad = lambda x, args: fun(x, *args)
	else:
		assert isinstance(jac, Callable), "Jacobian must be a Callable"
		call_obj_grad = lambda x, args: (fun(x, *args), jac(x, *args))
	it = 0
	grad = [np.inf]
	constraints_sat = True
	t = tqdm(range(maxiter))
	with np.printoptions(precision=3, floatmode="fixed", sign="+"):
		for it in t:
			optimizer.zero_grad()
			obj, grad = call_obj_grad(params, args)
			optimizer.step()
			params_ = params.detach().numpy().copy()
			if verbose:
				t.set_description(f"Step {it:>5}")
				t.set_postfix(params=np.asarray(params_), loss=np.asarray(obj).item(), grad=np.asarray(grad))
			steps.append(params_)
			if callback is not None:
				constraints_sat = callback(params.detach().numpy())
			if np.linalg.norm(grad) <= atol or not constraints_sat:
				break
	t.close()
	return np.array(steps)


## Implements Alg. 1 from https://people.maths.ox.ac.uk/hauser/hauser_lecture3.pdf
def optimize_trust_region(fun: Callable, x0, eta: tuple = (0.9, 0.1), gamma: float = 1.0):
	import trustregion

	trustregion
