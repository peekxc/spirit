#import "../math_ops.typ": * 
#import "../theorems.typ": *

= Computational Implications <sec:computational_imp>
== Exact computation <sec:lanczos_it>
As evidenced by @sec:spectral_sec, computing $hat(mu)_p^ast$ and $hat(beta)_p^ast$ may be reduced to computing eigenvalues of $p$-Laplacians. To do this efficiently, we employ the _Lanczos method_ @lanczos1950iteration, which estimates the eigenvalues of any symmetric linear operator $A$ via projection onto successive Krylov subspaces. Given a symmetric $A in bb(R)^(n times n)$ with eigenvalues $lambda_1 >= lambda_2 gt dots.h >= lambda_r gt 0$ and a vector $v eq.not 0$, the Lanczos method generates the triple $(K , Q , T)$: 

$ K & = lr([thin A^0 v divides A^1 v divides A^2 v divides dots.h divides A^(r - 1) v thin]) \
Q & = lr([thin q_1 , q_2 , dots.h , q_r thin]) arrow.l upright(q r) (K)\
T & = Q^T A Q $ 

where $K in bb(R)^(n times r)$ is the _Krylov matrix_ with respect to $(A , v)$, $Q in bb(R)^(n times r)$ is an orthogonal change-of-basis, and $T in bb(R)^(r times r)$ is symmetric tridiagonal _Jacobi matrix_. It is well known that $Q$ is a similarity transform, i.e. $T$ preserves the spectrum of $A$ and the problem of finding eigenvalues reduces to diagonalizing $T$.

The Lanczos method is often called a "matrix free" method due to its only prerequisite for computation being a matrix-vector product operator $v |-> A v$—$A$ need not necessarily be stored in memory explicitly. For this reason, the Lanczos method is often used as a low-memory option for computing eigenvalues. Indeed, due to its _three-term recurrence_ @simon1984analysis, the Lanczos method requires just three $O (n)$-sized vectors and a constant number of $O (n)$ vector operations to obtain $T$—neither $K$ nor $Q$ need be formed explicitly.


#lemma([@parlett1994we])[
	Given a symmetric rank-$r$ matrix $A in bb(R)^(n times n)$ whose matrix-vector operator $x |-> A x$ requires $O (eta)$ time and $O (nu)$ space, the Lanczos iteration computes $Lambda (A) = brace.l lambda_1, lambda_2, dots.h, lambda_r brace.r$ in $O lr((max brace.l eta, n brace.r dot.op r))$ time and $O lr((max brace.l nu, n brace.r))$ space, when computation is done in exact arithmetic.
] <lemma:exact_arith_lanczos>

Surprisingly, @lemma:exact_arith_lanczos suggests the time and space complexities of computing quantities that depend on $brace.l thin lambda_1, lambda_2, dots.h, lambda_r thin brace.r$ are lower for certain classes of matrices, provided $nu$ and $eta$ are sufficiently small. In general, the size of these variables depends on the structure and sparsity of the operators#footnote[For $n times n$ sparse matrices with an average of $z$ non-zeros per row and rank $r$, for example, @lemma:exact_arith_lanczos implies the Lanczos method has an expected $O (n z r)$ time and $O (n z)$ space complexities @golub2013matrix.] involved. In some cases, both $nu$ and $eta$ are $O(n)$; for example, the application $x |-> L x$ for the graph Laplacian $L = diff_1 diff_1^T$ is linear in the number of edges $abs(E)$ due to its graph structure. Though this result is well established for the graph Laplacian, it is not immediately clear whether a similar guarantee generalizes to combinatorial Laplacian operators derived from simplicial complexes—our next result affirms this.

#lemma[
	For any $p >= 0$ and simplicial complex $K$ with $n = abs(K^p)$ and $m = abs(K^(p + 1))$, if there exists a hash function $h:K^p -> lr([n])$ constructible in $O (m)$ time supporting $O (1)$ access time and $O (c)$ space, then there exists a two-phase algorithm for computing the product $x |-> L_p x$ in $O (m lr((p + 1)))$ time and $O (max lr((c , m)))$ space.
] <lemma:matvec_lap>

The algorithm and proof are given in @sec:appendix. From a practical perspective, many hash table implementations achieve expected $O (1)$ access time using only a linear amount of storage, and as $p >= 0$ is typically quite small—the operation $x |-> L x$ in practice exhibits $approx O (m)$ time and space complexities. We delegate more practical issues regarding the computation to @alg:lap_matvec. Combining @lemma:exact_arith_lanczos and @lemma:matvec_lap yields our main result for this section.

#proposition[
	For any constant $p >= 0$ and box $R = lr([a , b]) times lr([c , d]) subset Delta_+$, the persistent multiplicity function $mu_p^R (K)$ derived from a simplicial complex $K$ with $n_(a d) = abs(K_d^p )- abs(K_a^p)$ and $m_(a d) = abs(K_d^(p + 1))- abs(K_a^(p + 1))$, can be computed in exact arithmetic with the _Lanczos method_ in the following time and space complexities: 
	
	$ mu_p^R (K) eq^(text("time")) O(n_(a d) dot.op m_(a d)), quad mu_p^R (K) eq^(text("space")) O (max lr((n_(a d) , m_(a d)))) $ 
	
	In particular, when $R = lr((- oo , ast]) times lr([ast, + oo))$, $mu_p^R (K)$ has time and space complexities of $O(n m)$ and $O(max(n,m))$, respectively, where $n = abs(K^p)$ and $m = abs(K^(p+1))$.
] <prop:spectral_rank_complexity>

It's worth noting that the standard reduction-family of algorithms computes the $p$-th persistent homology of a filtration $K$ of dimension $p + 1$ and of size $N = abs(K )tilde.op O lr((abs(K^(p + 1))))$ in $Theta lr((N^3))$ time and $Theta lr((N^2))$ space. Interestingly, Chen and Kerber @chen2011output have shown that since the persistence diagram contains at most $N slash 2 = O (N)$ points, it may be constructed using at most $2 N - 1$ "$mu$-queries" (evaluations of $mu_p^R$) via a divide-and-conquer scheme on the index-persistence plane. Since both $abs(K^p)$ and $abs(K^(p + 1))$ are trivially bounded by $O (N)$, by @prop:spectral_rank_complexity, we may recover the same $O lr((N^3))$ time complexity of the reduction algorithm using only rank computations, and we improve the space complexity by a factor of $N$, though at the cost of not having immediate access to cycle representatives.

#remark[
	A similar result for computing _non-persistent_ Betti numbers of simplicial complexes over finite fields was given by Edelsbrunner and Parsa in @edelsbrunner2014computational, wherein the complexity of computing the Betti numbers of a 2-dimensional simplicial complex $K$ with $n$ vertices was shown to be $Omega (r lr((n , m)))$, where $r (n , m)$ is the complexity of computing the rank of an $n$-by-$n$ binary matrix with $m$ non-zero entries.
]

// == Finite-precision arithmetic <sec:lanczos_fp>
	// In the exact arithmetic model, the choice of Laplacian weights (and thus, the inner product @eq:inner_product_cochain) does not affect the rank computation: once the Krylov space becomes $A$-invariant, the iteration ends.
	// It is known the convergence of the Lanczos method depends on the relative differences in eigenvalues in the spectrum. For the purpose of rank estimation of a matrix $A$, one requires the Krylov dimension $r$ to satisfy $r >= rank(A)$, thus the convergence rate is dominated by the magnitude of the smallest positive eigenvalue $lambda_r$, sometimes called the _spectral gap_. 
// ]

== Randomized $lr((eta , epsilon.alt))$-approximation <sec:iterative_approx>

As in @parlett1994we, @prop:spectral_rank_complexity assumes an exact arithmetic computation model to simplify both the presentation of the theory and the corresponding complexity statements. In practice, finite-precision arithmetic introduces _both_ rounding and cancellation errors into the computation affect both the convergence and termination conditions of the Lanczos method, prohibiting its use practically. 
Though many improvements have been proposed throughout the decades (e.g. selective re-orthogonalization, implicit restarting, see @sorensen1995implicitly @lehoucq1998arpack for an overview), it is well-known that the Lanczos method is not efficient at accurately approximating eigenvalues on the interior of the spectrum.

Fortunately, it turns out accurately estimating eigenvalues is not necessary for estimating the rank. Recently, it has been shown that the Lanczos method is stable for matrix function approximation#footnote[Recall that if $A in bb(R)^(n times n)$ has eigenvalue decomposition $A = U Lambda U^T$, the matrix function $f (A) in bb(R)^(n times n)$ with respect to a function $f$ is $U f (Lambda) U^T$, where $f (Lambda)$ applies $f$ to each diagonal entry of $Lambda$.] even in the finite precision arithmetic model. The use of Lanczos for matrix function approximation is motivated by the fact that Lanczos expansion up to degree $k$ can _exactly_ apply any matrix polynomial $p$ with $op("degree") < k$ @musco2018stability:

$ p(A) q_1 = Q p(T) e_1 arrow.l.r.double quad f (A) x = norm(x) dot.op Q f (T) e_1 $ <eq:lanczos_mf>

In particular, Paige's A27 Lanczos variant (@paige1972computational), when executed up to degree $k$, has an error that is bounded by the uniform error the best degree-$p$ polynomial approximation to any bounded function $f$ with degree $p < k$ (see @musco2018stability for conditions). 
For general matrix functions $f(A)$, this implies that finite-precision Lanczos essentially matches the strongest known exact arithmetic bounds. 
The particular relevance of @eq:lanczos_mf to the computation of our proposed spectral $phi.alt$-approximation @eq:lowner is the fact that the quantity of interest $norm(Phi(cal(L)_p))_ast$ is expressible as a trace-norm, i.e. it may be computed via: 
// f = sgn_+ arrow.l.r.double tr(sgn_+(A)) = rank (A)
$	norm(Phi_tau (A))_ast = tr (Phi_tau (cal(L)_p)) = sum_(i=1)^n e_i^T Phi_tau (cal(L)_p) e_i
$
where ${e_1, dots, e_n}$ is an orthonormal basis. For this reason, we propose the use _stochastic Lanczos quadrature_ (SLQ) type methods @ubaru2016fast for estimating quantities of the form $tr (f (A))$. The simplest such estimator is the Girard-Hutchinson (GH) estimator: 

$ tr (f (A)) approx n / n_v sum_(j = 1)^(n_v) e_1^T f lr((T_k^((j)))) e_1 =
	n / n_v sum_(j = 1)^(n_v) lr((sum_(i = 1)^k tau_i^((j)) f lr((theta_i^((j)))))), 
	quad tau_i = lr([e_1^T y_i])^2 
$ <eq:gh_trace_estimator>

where $T_k^((j))$ represents the result applying Lanczos to the degree-$(k + 1)$ Krylov expansion of $lr((A , v^((j))))$ and $(theta_i , y_i)$ represent the Rayleigh-Ritz pairs associated with the tridiagonal eigendecomposition $T_k = Y Theta Y^T$. When $v^((j)) tilde.op cal(D)$ derives from a sub-Gaussian distribution satisfying $bb(E) lr([v^((j)) times.circle v^((j))]) = I$, the approximation @eq:gh_trace_estimator is known to be an unbiased estimator of $tr (f (A))$. Under mild assumptions (see @ubaru2016fast), if the function of interest $f:lr([a , b]) -> bb(R)$ is analytic on $lr([lambda_min, lambda_max])$, then for constants $epsilon.alt , eta in (0 , 1)$ the GH estimator $Gamma$ satisfies: 

$ Pr(abs(tr (f (A)) - Gamma) <= epsilon.alt abs(tr (f (A)))) >= 1 - eta $ 

In other words, we can achieve a relative $epsilon.alt$-approximation of $tr (f (A))$ with success probability $eta$ using on the order of $O lr((epsilon.alt^(- 2) log lr((eta^(- 1)))))$ evaluations of $e_1^T f (T_k) e_1$, where each $T_k$ is generated by applying degree-$(k + 1)$ Lanczos to pairs $(A , v)$ generated over random $v tilde.op cal(D)$. In @sec:limitations, we show how to extend this result to the persistence setting.

More recent work by @meyer2021hutchpp has shown that deflation techniques can reduce the number of matrix-vector evaluations needed down to $O( epsilon^(-1) dot sqrt( log (eta^(-1))) + log (eta^(-1)))$, though we note in @sec:limitations there are additional limitations to be aware of using these methods (e.g. a $O(m k)$ space complexity). 


== Apparent pairs optimization <sec:apparent-pairs-optimization>
One of the defining aspects of the above trace-based computation is that, unlike the reduction algorithm, execution does not require matrix decomposition. Unfortunately, one downside to this is that we lose access to certain computational shortcuts developed specifically for the reduction setting; it is not immediately clear whether persistence-specific optimizations like _clearing_ and _cohomology_ @dey2022computational have analogous shortcuts in the matrix-free setting. Nonetheless, one such optimization well known for clique filtrations—the identification of _apparent pairs_—does have a direct translation to the trace estimation setting. On some types of problems, this optimization alone enables us to discard 90-99% of the columns of $diff_p$ prior to any rank computations.

Apparent pairs (APs) are a class of persistence pairs which are already reduced in the filtration boundary matrix $diff (K)$. To motivate their use in our proposed trace-based computation, consider the following lemma:

#lemma[
	Let $diff_(p + 1) (K)$ denote the dimension $p + 1$ filtered boundary matrix obtained from the $R = diff V$ decomposition of a simplexwise filtration $(K , f)$. Then, the $p$-th persistence diagram $dgm_p (K)$ determines a partitioning of the columns of $diff_(p + 1) (K)$ into submatrices $diff_(p + 1)^-$ and $diff_(p + 1)^+$: 
	
	$ diff_(p + 1)^- = brace.l thin diff lr([sigma_j]):col_R (j) eq.not 0 , sigma_j in K^(p + 1) thin brace.r , quad diff_(p + 1)^+ = brace.l thin diff lr([sigma_j]):col_R (j) = 0 , sigma_j in K^(p + 1) thin brace.r $ 
	and so that: 
	$ rank(diff_(p + 1) (K)) = rank(diff_(p + 1)^- (K)) $
] <lemma:ap_kernel>

In other words, from a rank-based perspective, we may safely discard $p + 1$ simplices whose boundary chains lie in the kernel of $R$. Of course, if $diff_(p + 1)^-$ if obtained from the full decomposition $R = diff K$, then the rank of any submatrix of $diff$ is fully determined, leaving the utility of the above observation moot.

Fortunately, for simplexwise clique filtrations, we can use APs $(tau , sigma) in dgm_(p + 1) (K)$ to construct an approximation $hat(diff)_(p + 1)^- supset diff_(p + 1)^-$ efficiently. The main benefit of using APs here is that they can be readily identified based on a purely local condition, which is evident from their very definition:

#definition("Apparent Pair")[
	Given a simplexwise filtration $K_prec.eq$ of a complex $K$, a pair of simplices $(tau , sigma)$ of $K$ is called an _apparent pair_ of $K_prec.eq$ if (1) $tau$ is youngest facet of $sigma$, and (2) $sigma$ is the oldest cofacet of $tau$.
] <def:apparent_pair>

Equivalently, a pair $(tau , sigma)$ of $K$ is _apparent_ if all entries below or to the left of $(tau , sigma)$ in the filtered boundary matrix $diff$ are zero. Since any persistent pair $(tau , sigma) in dgm_p (K)$ corresponds to a pair of columns in $R_p$ and $R_(p + 1)$ (respectively), the boundary chain $diff lr([sigma])$ must lie in the kernel of $diff_(p + 1) V_(p + 1)$, and therefore can be safely discarded. Note that the set of APs does not depend on the coefficient field the homology of the complex is defined over, though they do depend on the simplexwise ordering.

Identifying an AP $(tau , sigma) in dgm_p (K)$ can be reduced to enumerating cofacets of $tau in K^p$. To do this efficiently, we follow the low-memory approach used in the popular software Ripser @bauer2021ripser, which restricts its computation to simplexwise filtrations induced by the _reverse colexicographical_ vertex order. In this setting, $p$-simplices $(thin v_(i_p), dots.h , v_(i_0) thin)$ satisfying $v_(i_p) gt dots.h gt v_(i_0)$ are mapped to integers $r in bracket.l 0 , C(n, p + 1))$---where $C(n, k)$ is the binomial coefficient---via the _combinatorial number system_; when the colexicographical order is used, the bijection is given by the sum $(thin v_(i_(p)), dots.h , v_(i_0) thin) |-> sum_(j = 1)^(p + 1) C(i_j , j)$, and cofacets are given by the following relation: 

$ ( v_(i_(p)), dots.h , v_(i_(k + 1)) , v_j , v_(i_k) , dots.h , v_(i_0) ) |-> sum_(l = k + 1)^(p) vec(i_l, l + 2) + vec(j, k + 1) + sum_(l = 0)^k vec(i_l, l + 1) $ <eq:colex_order>

By enumerating $j = n - 1 , dots.h , 0$ for $j in.not lr([n]) backslash brace.l i_p , dots.h , i_0 brace.r$, one recovers all of $n - (p + 1)$ cofacets of $(thin v_(i_p), dots.h , v_(i_0) thin)$ in reverse colexicographic vertex order. Cofacet enumeration in the colexicographical order is particularly efficient due to the fact that the left and right partial sums in @eq:colex_order can be maintained throughout the enumeration: assuming all $O lr((n dot.op (p + 1)))$ binomial coefficients are precomputed, finding all the cofacets of any $p$-simplex $tau$ requires just $O (n - p)$ additions in integer arithmetic.

The most prevalent subset of APs relevant to the clique-filtrations are those with zero persistence. In practice, zero persistence APs may be identified without enumerating all cofacets by using a "shortcut" involving a correspondence between zero-persistence APs and lexicographically minimal (maximal, respectively) facet (cofacet, respectively) pairs (see Proposition 3.12 in @bauer2021ripser). This "shortcut" is particularly useful due to the fact that zero persistence pairs comprise a large proportion of the apparent pairs—indeed, Theorem 3.10 in @bauer2021ripser shows that in dimension 1, the zero persistence pairs of a simplexwise refinement of the Vietoris-Rips filtration are _precisely_ the apparent pairs of the same filtration.
