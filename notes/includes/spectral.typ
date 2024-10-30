#import "../math_ops.typ": * 
#import "../theorems.typ": *

== Spectral relaxation and its implications <sec:spectral_sec>
Before introducing our proposed relaxation, it is instructive to examine the how traditional expressions of the persistent rank invariants compare to those from @cor:rank_reduction. Given a filtration $(K , f)$ of size $N = abs(K)$ with $f:K -> I$ defined over some index set $I$, its $p$-th persistent Betti number $beta_p^(a , b)$ at index $(a , b) in I times I$, is defined as follows: 

$
beta_p^(a,b) &= dim(Z_p (K_a) med slash med B_p (K_b)) \
&= dim(Z_p (K_a) med slash med (Z_p (K_a) sect B_p (K_b)))  \
&= dim(Z_p (K_a)) - dim(Z_p (K_a) sect B_p (K_b)) 
$ <eq:pbn>

Computationally, observe that @eq:pbn reduces to one nullity computation and one subspace intersection computation. While the former is easy to re-cast as a spectral computation, computing the latter typically requires obtaining bases via matrix decomposition. Constructing these bases explicitly using conventional @bhatia2013matrix or persistence-based @zomorodian2004computing @memoli2022persistent algorithms effectively#footnote[An alternative means to perform matrix reduction is through the PLU factorization (see @dey2022computational, section 5.1.1), which takes $O(n^omega)$ time where $omega in [2, 2.373)$ is the matrix-multiplication constant. However, this approach exploits neither the structure of the boundary matrix nor properties of persistence to accelerate the computation and is not used in practice.] requires $Omega lr((N^3))$ time and $Omega lr((N^2))$ space. As the persistence algorithm also exhibits $O lr((N^3))$ time complexity and completely characterizes $beta_p^(a , b)$ over _all_ values $(a , b) in I times I$, there is little incentive to compute $beta_p^(a , b)$ with such direct methods (and indeed, they are largely unused). Because of this, we will focus on expressions @eq:betti_four and @eq:mu_four throughout the rest of the paper.

=== Parameterized boundary operators <sec:param_boundary>
In typical dynamic persistence settings (e.g. @cohen2006vines), a decomposition $R = diff V$ of the boundary matrix $diff$ must be permuted and modified frequently to maintain a simplexwise order respecting $f_alpha$. In contrast, one benefit of the rank function is its permutation invariance: for any $X in bb(R)^(n times n)$ and permutation $P$ we have: $ rank (X) = rank lr((P^T X P)) $ This suggests persistent rank computations like those from @cor:rank_reduction need not maintain this ordering—as long as the constitutive boundary matrices the same non-zero pattern as their filtered counterparts, their ranks will be identical. In what follows, we demonstrate how exploiting this permutation invariance significantly simplifies the practical use of @eq:betti_four and @eq:mu_four in _parameterized_ settings.

Let $(K , f_alpha)$ denote parameterized family of filtrations of a simplicial complex of size $abs(K^p )= n$. Fix an arbitrary linear extension $lr((K , prec.eq))$ of the face poset of $K$. Define the $cal(A)$-_parameterized_ _boundary operator_ $hat(diff)_p (alpha) in bb(R)^(n times n)$ of $(K , f_alpha)$ as the $n times n$ matrix ordered by $prec.eq$ for all $alpha in cal(A)$ whose entries $(k , l)$ satisfy: 

$ diff_p (alpha) lr([k , l]) = cases(delim: "{", s_(k l) dot.op f_alpha (sigma_k) dot.op f_alpha (sigma_l) & quad text("if ") sigma_k in diff_p (sigma_l), 0 & quad text("otherwise")) $ <eq:param_boundary_matrix>

where $s_(k l) = sgn lr((lr([sigma_k]) , diff lr([sigma_l])))$ is the sign of the oriented face $lr([sigma_k])$ in $diff lr([sigma_l])$. Observe that @eq:param_boundary_matrix may be decoupled into a product of diagonal matrices $D_ast (f_alpha)$: 

$ diff_p (alpha) eq.delta D_p (f_alpha) dot.op diff_p lr((K_prec.eq)) dot.op D_(p + 1) (f_alpha) $ <eq:decouple> 

where $D_p (f_alpha)$ and $D_(p + 1) (f_alpha)$ are diagonal matrices whose non-zero entries are ordered by restrictions of $f_alpha$ to $K_prec.eq^p$ and $K_prec.eq^(p + 1)$, respectively. Clearly, $rank (diff_p (alpha)) = rank lr((diff_p lr((K_prec.eq))))$ when the diagonal entries of $D_p$ and $D_(p + 1)$ are strictly positive. Moreover, observe we may restrict to those "lower left" matrices from @lemma:rank via post-composing step functions $ macron(S)_a (x) = bb(1)_(x gt a)$ and $S_b (x) = bb(1)_(x <= b) (x)$ to $D_p$ and $D_(p + 1)$, respectively: 

$ hat(diff)_p^(a , b) (alpha) eq.delta D_p (macron(S)_a compose f_alpha) dot.op diff_p lr((K_prec.eq)) dot.op D_(p + 1) (S_b compose f_alpha) $  <eq:rank_equiv_param>

Though these step functions are discontinuous at their chosen thresholds $a$ and $b$, we may retain the element-wise continuity of @eq:decouple by exchanging them with clamped _smoothstep_ functions $cal(S):bb(R) -> lr([0 , 1])$ that interpolate the discontinuous step portion of $S$ along a fixed interval $(a , a + omega)$, for some $omega gt 0$ (see @fig:smoothstep).

The observations above collectively motivate our first relaxation. Without loss in generality, assume the orientation of the simplices $lr((K , prec.eq))$ is induced by the order on the vertex set $V$. To simplify the notation, we write $A^x = A^(ast , x)$ to denote the submatrix including all rows of $A$ and all columns of $A$ up to $x$.

#proposition[
	Given $(K , f_alpha)$, any rectangle $R = lr([a , b]) times lr([c , d]) subset Delta_+$, and $delta gt 0$ the number satisfying $a + delta lt b - delta$ from @eq:measure the $cal(A)$-parameterized invariants $beta_p^(a , b):cal(A) times K -> bb(N)$ and $mu_p^R:cal(A) times K -> bb(N)$ defined by:
	$ beta_p^(a, b) (alpha) = rank(D_p (S_a compose f_alpha)) - rank(hat(partial)_p^(a) (alpha)) - rank(hat(partial)_(p+1)^(b) (alpha)) + rank(hat(partial)_(p+1)^(a+delta, b)) (alpha) $ <eq:pbn_parameterized>
	$ mu_p^R (alpha) = rank(hat(partial)_(p+1)^(b+delta, c)) - rank(hat(partial)_(p+1)^(a+delta, c)) - rank(hat(partial)_(p+1)^(b+delta, d)) + rank(hat(partial)_(p+1)^(a+delta, d)) $ <eq:mu_parameterized>
	yield the correct quantities $mu_p^R(K, f_alpha) = card lr(dgm_p (f_alpha)|_R)$ and $beta_p^(a,b) = dim (H_p^(a,b)(K, f_alpha))$ for all $alpha in cal(A)$. 
] <prop:mu_betti_1>
		
// For completeness, a proof of @prop:mu_betti_1 is given in the appendix. 
Note that in @eq:rank_equiv_param, we write $diff_p lr((K_prec.eq))$ (as opposed to $diff_p (K , f)$) to emphasize $diff_p lr((K_prec.eq))$ is ordered according to a fixed linear ordering $lr((K , prec.eq))$. The distinction is necessary as evaluating the boundary terms from @cor:rank_reduction would require $diff$ to be explicitly filtered in the total ordering induced by $f_alpha$—which varies in $cal(A)$—whereas the expressions obtained by replacing the constitutive terms in @eq:betti_four and @eq:mu_four with @eq:pbn_parameterized and @eq:mu_parameterized, respectively, requires no such explicit filtering.

#corollary[
	Given a boundary matrix $hat(diff)_p (alpha) in bb(R)^(n times n)$ constructed at time $alpha in bb(R)$ from a parameterized family of filtrations $(K , f_alpha)$ with linear extension $prec.eq$, the time complexity of constructing $hat(diff)_p (alpha prime)$ for any other $alpha prime eq.not alpha$ is $O lr((max lr((abs(K^p ), abs(K^(p + 1))))))$, assuming the evaluation of $f_alpha (tau)$ is $O (1)$ for every simplex $tau in K$.
]

== Parameterized Laplacians <sec:laplacian_theory2>
For generality's sake, it is important to make the class of expressions for $beta_p^ast$ and $mu_p^ast$ as large as possible. Since we are only concerned with homology over $bb(R)$, we may exploit another identity of the rank function which is only applicable to zero characteristic fields: 

$ rank (X) = rank lr((X X^T)) = rank lr((X^T X)) , quad text("for all ") X in bb(F)^(n times m) $ <eq:rank_invariance_adjoint>

In the context of boundary operators, note that $diff_1 diff_1^T$ is the well known _graph Laplacian_---more generally, @eq:rank_invariance_adjoint suggests we can readily express $beta_p^ast (alpha)$ and $mu_p^ast (alpha)$ using the ranks of combinatorial $p$-Laplacians.#footnote[By convention, we define $diff_p = 0$ for all $p <= 0$.]

Following the seminal results from Horak and Jost @horak2013spectra, there are three natural ways to define $p$-Laplacians over a fixed simplicial complex $K$: the _up_-Laplacian $L_p^up (K)$, the _down_-Laplacian $L_p^dn (K)$, and their sum, which we refer to as the _combinatorial_ Laplacian $Delta_p (K)$: 

$ Delta_p (K) = 
  underbrace(partial_(p+1) compose partial_(p+1)^T, L_p^up) + 
	underbrace(partial_(p)^T compose partial_(p), L_p^dn) 
$ <eq:comb_lap>

All three operators $Delta_p$, $L_p^(up)$, and $L_p^(dn)$ are symmetric, positive semi-definite, and compact @memoli2022persistent—moreover, the #emph[non-zero] multisets $Lambda lr((L_p^(up)))$ and $Lambda lr((L_(p + 1)^(dn)))$ are equivalent, implying they must have identical ranks (see Theorem 2.2 and 3.1 of @horak2013spectra). Thus, for rank computations, it suffices to consider only one of them.

Let $(K , f_alpha)$ denote a parameterized family of filtrations of a simplicial complex $K$ equipped with a fixed but arbitrary linear extension $prec.eq$ of its face poset and fixed orientations $s (sigma)$ inherited from the total order on the vertex set $lr((V , prec.eq))$. Without loss of generality, we define the weighted $p$ up-Laplacian $cal(L)_p eq.delta L_p^up$ at index $(a , b)$ as follows: 

$ cal(L)_(p , prec.eq)^(a , b) (alpha) & eq.delta D_p (macron(S)_a compose f_alpha) dot.op diff_(p + 1) lr((K_prec.eq)) dot.op D_(p + 1) (S_b compose f_alpha) dot.op diff_(p + 1)^T lr((K_prec.eq)) dot.op D_p (macron(S)_a compose f_alpha) $ <eq:laplacian_decouple>

where $D_p (f)$ denotes a diagonal matrix whose entries represent the application of $f$ to the $p$-simplices of $K$. As in @eq:rank_equiv_param, fixing step function $S_a$ and $macron(S)_b$ at values $a , b in bb(R)$ yields operators whose ranks correspond to the ranks of certain "lower-left" submatrices of the corresponding full boundary matrix $diff$ of $(K , f)$. In particular, if $R = diff V$ is the decomposition of $(K , f_alpha)$ for some fixed choice of $alpha in cal(A)$, then for any pair $(a , b) in Delta_+$ there exists indices $i = sum_(sigma in K) (macron(S)_a compose f_alpha) (sigma)$ and $j = sum_(sigma in K) (S_b compose f_alpha) (sigma)$ such that: 

$ rank lr((R_(p + 1)^(i , j))) = rank lr((diff_(p + 1)^(i , j))) = rank lr((hat(diff)_(p + 1)^(a , b))) = rank lr((lr((hat(diff)_(p + 1)^(a , b))) lr((hat(diff)_(p + 1)^(a , b)))^T)) = rank lr((cal(L)_(p , prec.eq)^(a , b))) $ 

where the second last equality uses the identity $rank (X) = rank lr((X^T X))$, which is true when $X in bb(F)^(n times m)$ has coefficients in a zero-characteristic field $bb(F)$. This confirms that we may substitute any of the parameterized boundary operators used in @prop:mu_betti_1 with weighted Laplacian operators $diff_(p + 1)^ast |-> cal(L)_p^ast$ equipped with the appropriate down- and up-step functions $S_ast$ and $macron(S)_ast$, respectively.

It is worth noting that composing step functions $macron(S)_a , S_b:bb(R) -> lr([0 , 1])$ with $f_alpha$ is equivalent to endowing a _weight function_ $w:K -> (0 , + oo)$ on a subset $K_(a , b) subset.eq K$, in the sense described by @memoli2022persistent. In particular, if $w_p$ denotes the restriction of $w$ to $K^p$, then $w_p$ defines an inner product $angle.l thin dot.op , dot.op thin angle.r_(w_p)$ on space of $p$-chains $C_p (K , bb(R))$ given by: 

$ angle.l lr([sigma]) , lr([sigma prime]) angle.r_(w_p) eq.delta delta_(sigma sigma prime) dot.op (w_p (sigma))^(- 1) , quad forall thin sigma , sigma prime in K^p $ <eq:inner_product_chain>

where $delta_(sigma sigma prime) = cal(1) lr((sigma = sigma prime))$ is the indicator function on $sigma$. Moreover, this inner product $angle.l lr([sigma]) , lr([sigma prime]) angle.r_(w_p)$ on $C_p (K , bb(R))$ induces an inner product:

$ angle.l.double f, g angle.r.double_(w) = sum_(sigma in K^p) f([sigma]) g([sigma]) w(sigma), 
	quad text("for all ") f,g in C^p (K)
$<eq:inner_product_cochain>

In these sense above, @eq:laplacian_decouple is simply choosing a particular inner product on the space of $p$-cochains.

// The insight provided by the inner product perspective is crucial for the validity of our method, as it is in part what allows use to generalize from simplicial boundary operators to $p$-th combinatorial Laplacians. Indeed, the _combinatorial Hodge Theorem_ establishes the equivalence between @eq:pers_homology and @eq:betti_four in the case $a = b$, and is often attributed as the connection by which one extends the graph Laplacian to more general operators (i.e. for $p >= 1$). Deeper exploration of these connections falls beyond the scope of this work; we refer the interested reader to @memoli2022persistent and references within for an overview.

#remark[
	One may interpret the action of sending a subset $S subset.eq K$ of $p$-simplices to $0$ as a restriction of $K$ to a sub-complex $L = K backslash S$, which suggests e.g. @eq:pbn and @eq:laplacian_decouple could be alternatively defined using the inclusion maps $L arrow.r.hook K$ between _simplicial pairs_ $(L , K)$, as in @memoli2022persistent.
]

== Spectral rank relaxation <sec:spectral_relax>
Under mild assumptions on $f_alpha$, the entries of the boundary operators from @eq:rank_equiv_param are continuous functions of $alpha$ when $S$ is substituted appropriately with smoothstep functions. In contrast, the quantities from @prop:mu_betti_1 are by definition discontinuous functions, as they are integer-valued due to the rank function. To circumvent this issue, we consider the spectral characterization of the rank function: 

$ rank (X) = sum_(i = 1)^n sgn_+ (sigma_i (X)), 
	quad quad 
	sgn_+ (x) = cases(med 1 & quad text("if ") x gt 0, med 0 & quad text("otherwise"))
$ <eq:rank_def>

In the above, $brace.l sigma_i brace.r_(i = 1)^n$ are the singular values $Sigma = diag lr((brace.l sigma_i brace.r_(i = 1)^n))$ from the singular value decomposition (SVD) $X = U Sigma V^T$ of $X in bb(R)^(n times m)$, and $sgn_+:bb(R) -> brace.l 0 , 1 brace.r$ is the one-sided sign function. As the singular values vary continuously under perturbations in $X$ @bhatia2013matrix, it is clear the discontinuity in @eq:rank_def manifests from the one-sided sign function—thus, a natural approach to relaxing @eq:rank_def is to first relax the $sgn_+$ function.

Our approach follows the seminal work of Mangasarian et al. @mangasarian1994class. Let $p:bb(R)_+ -> bb(R)_+$ denote a continuous density function and $nu:bb(R)_+ -> bb(R)_+$ is a continuous increasing function satisfying $nu (0) = 0$. One way to approximate the $sgn_+$ function is to integrate $tau$-smoothed variations $hat(delta)$ of the Dirac delta measure $delta$:


$ phi.alt(x, tau) = integral_(-infinity)^x hat(delta)(z, tau) op("dz"), quad hat(delta)(z, tau) = 1 / nu(tau) dot p(z / nu(tau)), quad forall z >= 0, tau > 0 $ <eq:phi>

In contrast to the $sgn_+$ function, if $p$ is continuous on $bb(R)_+$ then $phi.alt lr((dot.op , tau))$ is continuously differentiable on $bb(R)_+$, and if $p$ is bounded above on $bb(R)_+$, then $phi.alt lr((dot.op , tau))$ is globally Lipshitz continuous on $bb(R)_+$. Moreover, varying $tau in bb(R)_+$ in @eq:phi yields an $tau$-parameterized family of continuous $sgn_+$ relaxations $phi.alt:bb(R)_+ times bb(R)_(+ +) -> bb(R)_+$, where $tau gt 0$ controls the accuracy of the relaxation.

Many properties of the sign approximation from @eq:phi extend naturally to the rank function when substituted appropriately via @eq:rank_def. In particular, pairing $X = U Sigma V^T$ with a scalar-valued $phi.alt$ that is continuously differentiable at every entry $sigma$ of $Sigma$ yields a corresponding #emph[Löwner operator] $Phi_tau$ @bi2013approximation:

$ Phi_tau (X) eq.delta sum_(i = 1)^n phi.alt (sigma_i , tau) u_i v_i^T $ <eq:lowner>

// #definition([Spectral $phi.alt$-approximation])[
// 	Given $X in bb(R)^(n times m)$ with SVD $X = U Sigma V^T$, a fixed $tau gt 0$, and any choice of $phi.alt:bb(R)_+ times bb(R)_(+ +)$ satisfying @eq:phi, define the #emph[spectral $phi.alt$-approximation] $Phi_tau (X)$ of $X$ as: 
// 	$ Phi_tau (X) eq.delta sum_(i = 1)^n phi.alt (sigma_i , tau) u_i v_i^T $
// 	where $u_i$ and $v_i$ are the $i$th columns of $U$ and $V$, respectively.
// ]<def:phi_approx>

Note that when $X$ is positive semi-definite, @eq:lowner may be interpreted as a particular choice of _matrix function_ $f (X) eq.delta U f (Lambda) U^T$ from the matrix function calculus perspective @bhatia2013matrix. Unlike general matrix functions, however, the restrictions on $phi.alt$ @eq:phi grants the operators $Phi_tau$ a variety of attractive properties related to approximation, monotonicity, and differentiability.

#proposition([Bi et al. @bi2013approximation])[
	The operator $Phi_tau:bb(R)^(n times m) -> bb(R)^(n times m)$ defined by @eq:lowner satisfies:

	+ #emph[For any $tau >= 0$, the Schatten-1 norm $norm(Phi_tau (X))_ast$ of $Phi_tau (X)$ is given by $sum_(i = 1)^n phi.alt (sigma_i , tau)$]

	+ #emph[For any $tau prime >= tau$, $norm(Phi_(tau prime) (X))_ast <= norm(Phi_tau (X))_ast$ for all $X in bb(R)^(n times m)$.]

	+ #emph[For any given $X in bb(R)^(n times m)$ with rank $r = rank (X)$ and positive singular values $Lambda (X) = brace.l sigma_1 , sigma_2 , dots.h , sigma_r brace.r$: 
	
	$ 0 <= r - norm(Phi_tau (X))_ast <= r dot.op lr((1 - phi.alt (sigma_r , tau))) $ 
	
	Moreover, if $tau$ satisfies $0 lt tau <= sigma_r slash r$, then $r - norm(Phi_tau (X))$ is bounded above by a constant $c_phi.alt (r) >= 0$.]

	+ #emph[$norm(Phi_tau (X))_ast$ is globally Lipshitz continuous and semismooth#footnote[#emph[Here, "semismooth" refers to the existence certain directional derivatives in the limit as $tau -> 0^+$, see @bhatia2013matrix @bhatia2013matrix.]] on $bb(R)^(n times m)$.]
] <prop:operator_props> 

#figure([#image("../images/cont_relax.png", width: 90%)],
	placement: top, 
	caption: [
		From left to right—the $ell_1$ (red) norm and $ell_0$ (black) pseudo-norm are shown on the interval $lr([- 1 , 1])$; the relaxation $phi.alt lr((dot.op , tau))$ from @eq:tikhonov_sf at various values of $tau gt 0$ (red) and at $tau = 0$ (black); the step function $S_i (x)$ from @eq:rank_equiv_param; the smoothstep relaxation $cal(S)_i^omega$ discussed in @sec:param_boundary.
	]
) <fig:smoothstep>

Noting property (4), since the sum Lipshitz functions is also Lipshitz, it is easy to verify that replacing the rank function in all of the constitutive terms from @prop:mu_betti_1 yields Lipshitz continuous functions whenever the filter function $f_alpha$ is itself Lipshitz and the step functions from @eq:rank_equiv_param are smoothed ($omega gt 0$).

#remark[
	Though $Phi_tau$ is a continuously differentiable operator#footnote[It may be shown to be twice continuously differentiable at $X$ if $phi.alt$ is twice-differentiable at each $sigma_i (X)$, see @ding2018spectral] in $bb(R)^(n times m)$ for any $tau gt 0$, its Schatten-1 norm $norm(Phi_tau (X))_ast$ is only directionally differentiable everywhere on $bb(R)^(n times m)$ in the Hadamard sense, due to Proposition 2.2(d-e) of @bi2013approximation. $norm(Phi_tau (X))_ast$ is differentiable on the positive semi-definite cone $bb(S)_+^n$.
]

#strong[Interpretation \#1:] In many applications, it is common to regularize an ill-posed objective function to encourage simpler solutions or to prevent overfitting. For example, the classical least-squares approach to solving the linear system $A x = b$ is often augmented with the _Tikhonov regularization_ (TR) for some $tau gt 0$: 

$ x_tau^ast = "arg min"_(x in bb(R)^n) norm(A x - b)^2 + tau norm(x)^2 = lr((A^T A + tau I))^(- 1) A^T b $ <eq:tikhonov>

When $tau = 0$, one recovers the standard $ell_2$ minimization, whereas when $tau gt 0$ solutions $x_tau^ast$ with small norm are favored. Similarly, by parameterizing $phi.alt$ by $nu (tau) = sqrt(tau)$ and $p (x) = 2 x lr((x^2 + 1))^(- 2)$, one obtains via  @eq:phi: 

$ phi.alt (x , tau) = integral_0^z hat(delta) (z , tau) d z = 2 / tau integral_0^z z dot.op ( (z slash sqrt(tau))^2 + 1 )^(- 2) d z = frac(x^2, x^2 + tau) $ <eq:tikhonov_sf>

By substituting $sgn_+ |-> phi.alt$ and composing with the singular value function @eq:lowner, the corresponding spectral rank approximation reduces#footnote[See Theorem 2 of @zhao2012approximation for a proof of the second equality.] to the following _trace_ formulation: 

$ norm(Phi_tau (A))_ast = sum_(i = 1)^n frac(sigma_i (A)^2, sigma_i (A)^2 + tau) = tr lr([lr((A^T A + tau thin I))^(- 1) A^T A]) $ <eq:tikhonov_1>

The relaxation level $tau$ may be thought of as a regularization term that preferences smaller singular values: larger values smooth out $norm(Phi_tau lr((dot.op)))_ast$ by making the pseudo-inverse less sensitive to perturbations, whereas smaller values lead to a more faithful#footnote[This can be seen directly by @eq:tikhonov as well, wherein increasing $tau$ lowers the condition number of $A^T A + tau I$ monotonically, signaling a tradeoff in stability at the expense of accuracy.] approximations of the rank. In this sense, we interpret the quantities obtained by applying @eq:lowner to the terms from @prop:mu_betti_1 as _regularized RI approximation_. 

#strong[Interpretation \#2:] In shape analysis applications, matrix functions are often used to simulate diffusion processes on meshes or graphs embedded in $bb(R)^d$ to obtain information of about their geometry. For example, consider a weighted graph $G = (V , E)$ with $n = abs(V)$ vertices with graph Laplacian $L_G = diff_1 diff_1^T$. The _heat_ of every vertex $v (t) in bb(R)^n$ as a function of time $t >= 0$ is governed by $L_G$ and the _heat equation_: 

$ v prime (t) = - L_G v (0) quad arrow.l.r.double quad L_G dot.op u (x , t) = - diff u (x , t) slash diff t $ <eq:heat_eq>

To simulate a diffusion process on $G$ from an initial distribution of heat $v (0) in bb(R)^n$, it suffices to construct the _heat kernel_ $H_t eq.delta exp lr((- t dot.op L_G))$ via the spectral decomposition $L_G = U Lambda U^T$ of $L_G$: 

$ v (t) = H_t thin v (0) , text(" where ") H_t = sum_(i = 1)^n e^(- t lambda_i) thin u_i thin u_i^T $ <eq:heat_kernel>

The heat kernel is invariant under isometric deformations, stable under perturbations, and is known to contain multiscale geometric information due to its close connection to geodesics. As is clear from @eq:heat_kernel, it is also a matrix function. Now, consider @eq:phi with $nu (tau) = tau$ and $p (lambda) = exp (- lambda_+)$ where $x_+ = max (x , 0)$: 

$ phi.alt (lambda , tau) = integral_0^z hat(delta) (z , tau) d z = 1 / tau integral_0^z exp(- z slash tau) d z = 1 - exp (- lambda slash tau) , quad text(" for all ") lambda >= 0 $  <eq:heat_sf>

In the context of diffusion, observe the parameter $tau$ is inversely related diffusion time (i.e. $t = 1 slash tau$) and that as $t -> 0$ (or $tau -> oo$) the expression $1 - exp (- lambda slash tau)$ approaches the $sgn_+$ function on the interval $bracket.l 0 , oo paren.r$. As above, substituting $phi.alt$ appropriately into @eq:lowner again yields an equivalent trace expression:

$ norm(Phi_tau (L_G))_ast = sum_(i = 1)^n 1 - exp (- lambda_i slash tau) = n - tr lr([H_(1 slash tau)]) $ <eq:heat_trace>

The heat kernel $H_t$ has been shown to fully characterize shapes up to isometry, motivating the creation of various geometric signatures, such as the Heat Kernel Signature (HKS) and the Heat Kernel Trace. In this sense, we interpret a spectral rank relaxation using @eq:heat_sf as a _geometrically informative RI approximation_.