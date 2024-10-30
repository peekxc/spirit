import os
import numpy as np


def shakeandbake(A: np.ndarray, b: np.ndarray, ns: int = 100, R_HOME: str = None) -> np.ndarray:
	R_HOME = str(R_HOME) if R_HOME is not None else "/Library/Frameworks/R.framework/Resources"
	os.environ["R_HOME"] = os.environ.get("R_HOME", R_HOME)
	import rpy2.robjects as robjects
	import rpy2.robjects.numpy2ri

	rpy2.robjects.numpy2ri.activate()

	r_script = """
	function(A, rhs, ns) {
		library(hitandrun)
		constr <- list(
			constr = A,
			dir = rep_len("<=", dim(A)[1]),
			rhs = rhs
		)
		X <- hitandrun::shakeandbake(constr, n.samples=ns)
		return(X)
	}
	"""
	r_function = robjects.r(r_script)
	X = r_function(A, b, ns)
	return X
