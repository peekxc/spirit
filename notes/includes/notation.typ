#import "../math_ops.typ": * 
#import "../theorems.typ": *

= Notation & Background <sec:background_notation>
A _simplicial complex_ $K subset.eq cal(P) (V)$ over a finite set $V = brace.l v_1 , v_2 , dots.h , v_n brace.r$ is a collection of simplices $brace.l thin sigma:sigma in cal(P) (V) thin brace.r$ such that $tau subset.eq sigma in K => tau in K$. A #emph[$p$-simplex] $sigma subset.eq V$ is a set of $p + 1$ vertices, the collection of which is denoted as $K^p$. An #emph[oriented $p$-simplex] $lr([sigma])$ is an ordered set $lr([sigma]) = (- 1)^(abs(pi)) lr([v_(pi (1)) , v_(pi (2)) , dots.h , v_(pi (p + 1))])$, where $pi$ is a permutation on $lr([thin p + 1 thin]) = brace.l thin 1 , 2 , dots.h , p + 1 thin brace.r$ and $abs(pi)$ the number of its inversions. The #emph[$p$-boundary] $diff_p lr([sigma])$ of an oriented $p$-simplex $lr([sigma]) in K$ is defined as the alternating sum of its oriented co-dimension 1 faces, which collectively for all $sigma in K^p$ define the $p$-th _boundary matrix_ $diff_p$ of $K$: 

$ diff_p lr([i , j]) eq.delta cases(delim: "{", 
	lr(
		(- 1))^(s_(i j)) & quad sigma_i in diff lr([sigma_j]), 
		0 & quad text("otherwise")
	) 
	med , quad quad 
	diff_p lr([sigma]) eq.delta sum_(i = 1)^(p + 1) (- 1)^(i - 1) lr([v_1, dots.h, v_(i - 1), v_(i + 1), dots.h v_(p + 1)]) 
$ <eq:alt_sum>

where $s_(i j) = sgn lr((lr([sigma_i]) , diff lr([sigma_j])))$ records the orientation. Extending @eq:alt_sum to all simplices in $sigma in K$ for all $p <= dim (K)$ yields the _full boundary matrix_ $diff$. With a small abuse in notation, we use $diff_p$ to denote both the boundary operator and its ordered matrix representative. When it is not clear from the context, we will clarify which representation is intended.

Generalizing beyond simplices, given a field $bb(F)$, an #emph[oriented $p$-chain] is a formal $bb(F)$-linear combination of oriented $p$-simplices of $K$ whose boundary $diff_p lr([c])$ is defined linearly in terms of its constitutive simplices. The collection of $p$-chains under addition yields an $bb(F)$-vector space $C_p (K)$ whose boundaries $c in diff_p lr([c prime])$ satisfying $diff_p lr([c]) = 0$ are called _cycles_. Together, the collection of $p$-boundaries and $p$-cycles forms the groups $B_p (K) = upright(I m) thin diff_(p + 1)$ and $Z_p (K) = upright(K e r) thin diff_p$, respectively. The quotient space $H_p (K) = Z_p (K) slash B_p (K)$ is called the #emph[$p$-th homology group of $K$] with coefficients in $bb(F)$ and its dimension $beta_p$ is the #emph[$p$-th Betti number] of $K$.

A _filtration_ is a pair $(K , f)$ where $f:K -> I$ is a _filter function_ over an index set $I$ satisfying $f (tau) <= f (sigma)$ whenever $tau subset.eq sigma$, for any $tau , sigma in K$. For every pair $(a , b) in I times I$ satisfying $a <= b$, the sequence of inclusions $K_a subset.eq dots.h subset.eq K_b$ induce linear transformations $h_p^(a , b):H_p (K_a) -> H_p (K_b)$ at the level of homology. When $bb(F)$ is a field, this sequence of homology groups uniquely decompose $(K , f)$ into a pairing $(sigma_a , sigma_b)$ demarcating the evolution of homology classes @zomorodian2004computing: $sigma_a$ marks the creation of a homology class, $sigma_b$ marks its destruction, and the difference $abs(a - b)$ records the lifetime of the class, called its _persistence_. The persistent homology groups are the images of these maps and the persistent Betti numbers are their dimensions: 

$ H_p^(a , b) = cases(delim: "{", H_p (K_a) & quad a = b, Im thin h_p^(a, b) & quad a lt b) 
med , quad quad 
beta_p^(a , b) = cases(delim: "{", beta_p (K_a) & quad a = b, dim lr((H_p^(a, b))) & quad a lt b) 
$<eq:pers_homology>
 
For a fixed $p >= 0$, the collection of persistent pairs $(a , b)$ together with unpaired simplices $(c , oo)$ form a summary representation $dgm_p (K , f)$ called the #emph[$p$-th persistence diagram of $(K , f)$]. Conceptually, $beta_p^(a , b)$ counts the number of persistent pairs lying inside the box $paren.l - oo , a thin bracket.r times (thin b , oo)$—the number of persistent homology groups born at or before $a$ that died sometime after $b$. When a given quantity depends on fixed parameters that are irrelevant or unknown, we use an asterisk. Thus, $H_p^ast (K)$ refers to any homology group of $K$.

We will at times need to generalize the notation given thus far to the _parameterized_ setting. Towards this end, for some $cal(A) subset.eq bb(R)^d$, we define an #emph[$cal(A)$-parameterized filtration] as a pair $(K , f_alpha)$ where $K$ is a simplicial complex and $f:K times cal(A) -> bb(R)$ an $cal(A)$-parameterized filter function satisfying: 

$ f_alpha (tau) <= f_alpha (sigma) med forall med tau subset.eq sigma in K text(" and ") f_alpha (sigma) text(" is continuous in ") alpha in cal(A) text(" for every ") sigma in K $

Intuitively, when $cal(A) = bb(R)$, one can think of $alpha$ as a _time_ parameter and each $f_alpha (sigma)$ as tracing a curve in $bb(R)^2$ parameterized by $alpha$. Examples of parameterized filtrations include:

- (Constant filtration) For a filter $f:K -> bb(R)$, note $(K , f_alpha)$ generalizes the "static" notion of a filtration $(K, f)$ in the sense one may declare $f_alpha (sigma) = f (sigma)$ for all $alpha in cal(A)$ and all $sigma in K$.

- (Dynamic Metric Spaces) For a set $X$, let $gamma_X = lr((X , d_X lr((dot.op))))$ denote a dynamic metric space @kim2021spatiotemporal, where $d_X lr((dot.op)):bb(R) times X times X$ denotes a time-varying metric. For any fixed $K subset cal(P) (X)$, the pair $(K , f_alpha)$ obtained by setting $f_alpha (sigma) = max_(x , x prime in sigma) d_X (alpha) (x , x prime)$ recovers the notion of a #emph[time-varying Rips filtration].

- (Interpolating filtrations) For $f,g: K -> bb(R)$ filters over $K$, a natural family of filtrations $(K , h_alpha)$ is obtained by choosing a homotopy $h : bb(R) times [0, 1] -> bb(R)$ satisfying $h_0 = f$ and $h_1  = g$.
//  _convex combinations_ of $f$ and $g$ e.g. $h_alpha = (1 - alpha) f + alpha g$ for all $alpha in lr([0 , 1])$, i.e.. More generally, any homotopy $$ between filtrations may be used. 

- (Fibered barcode) For a 2-d persistence module $M$, a common invariant of interest is the _fibered barcode_, which is the collection of barcodes of $1$-d affine slices of $M$. These affine slices are themselves a 2-parameter family spanning the collection of lines in $bb(R)^2$ with non-negative slope.