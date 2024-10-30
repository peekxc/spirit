#import "../math_ops.typ": * 
#import "../theorems.typ": *

= Introduction <sec:intro>
Persistent homology @edelsbrunner2000topological (PH) is the most widely deployed tool for data analysis and learning applications within the topological data analysis (TDA) community. 
Persistence-related pipelines often follow a common pattern: given a data set $X$ as input, construct a simplicial complex $K$ and an order-preserving function $f:K -> bb(R)$ such that useful topological/geometric information may be gleaned from its _persistence diagram_—a multiset summary of $f$ formed by pairs $(a , b) in bb(R)^2$ exhibiting non-zero _multiplicity_ $mu_p^(a , b) in bb(Z)_+$ @cerri2013betti: 

$ 
	dgm_p (K, f) &= { (a,b) : mu_p^(a,b) eq.not 0 } \
	mu_p^(a,b) &= min_(delta > 0) (beta_p^(a+delta, b-delta) - beta_p^(a+delta, b+delta)) - (beta_p^(a-delta, b-delta) - beta_p^(a-delta, b+delta)) 
$ <eq:dgm>

The surprising and essential quality of persistence is that these pairings exist, are unique, and are stable under additive perturbations @cohen2005stability. Whether for shape recognition @chazal2009gromov, dimensionality reduction @scoccola2023fibered, or time series analysis @perea2016persistent, persistence is the de facto connection between homology and the application frontier.

#figure([#image("../images/spectral_relax_size_func.png", width: 98%)],
	placement: top,
  caption: [
    (left) A function $f:bb(R) -> bb(R)$ and its corresponding persistence diagram obtained by filtering $f$ by its sublevel sets. (right) Two spectral interpolations of the rank function to certain Laplacian norms.
  ]
) <fig:overview>

Though theoretically sound, diagrams suffer from many practical issues: they are sensitive to outliers, far from injective, and expensive---both to compute _and_ compare. Towards ameliorating these issues, practitioners have equipped diagrams with additional structure by way of maps to function spaces; examples include persistence images @adams2017persistence, persistence landscapes @bubenik2015statistical, and template functions @perea2022approximating. Tackling the issue of injectivity, Turner et al. @turner2014persistent propose an injective shape statistic of directional diagrams associated to a data set $X subset bb(R)^d$, sparking both an inverse theory for persistence and a mathematical foundation for metric learning. Despite the potential these extensions have in learning applications, scalability issues due to high algorithmic complexity remain. Indeed, this issue is compounded in the parameterized setting, where adaptations of the persistence computation has proven non-trivial @piekenbrock2021move.

We seek to shift the computational paradigm on persistence while retaining its application potential: rather than following a construct-then-vectorize approach, we devise a spectral method that performs both steps, simultaneously and approximately. Our strategy is motivated both by a technical observation that suggests advantages exist for the rank invariant computation (@sec:betti_derivation) and by measure-theoretic results on $bb(R)$-indexed persistence modules @cerri2013betti @chazal2016structure, which generalize @eq:dgm to rectangles $R = lr([a , b]) times lr([c , d]) subset bb(R)^2$: 

$ mu_p^R (K, f) = card(dgm_p (K, f)|_(R)) = beta_p^(b,c) - beta_p^(a,c) - beta_p^(b,d) + beta_p^(a,d) $ <eq:measure>

Notably, our approach not only avoids explicitly constructing diagrams, but is also #emph[matrix-free], circumventing the reduction algorithm from @edelsbrunner2022computational entirely. Additionally, the relaxation is computable exactly in linear space and quadratic time, requires no complicated data structures or maintenance procedures to implement, and can be iteratively $lr((1 plus.minus epsilon.alt))$ approximated in effectively linear time in practice for large enough $epsilon.alt gt 0$. 

#strong[Contributions:] Our primary contribution is the introduction of several families of spectral approximations to the rank invariants—$mu_p$ and $beta_p$—all of which are Lipshitz continuous, and differentiable on the positive semi-definite cone (@prop:operator_props). By a reduction to spectral methods for Laplacian operators (@sec:laplacian_theory2), we also show these approximations are computable in $O (m)$ memory and $O (m n)$ time, where $n , m$ are the number of $p , p + 1$ simplices in $K$, respectively (@prop:spectral_rank_complexity). Moreover, both relaxations admit iterative $lr((1 plus.minus epsilon.alt))$-approximation schemes (@sec:iterative_approx), recovering both invariants $epsilon.alt$ is made small enough.  
