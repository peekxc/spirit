#import "@preview/lovelace:0.3.0": *
#import "../math_ops.typ": * 
#import "../theorems.typ": *

= Appendix <sec:appendix>
== Proofs <sec:proofs>

#proof([of @lemma:rank @dey2022computational])[
	The Pairing Uniqueness Lemma @edelsbrunner2000topological asserts that if $R = diff V$ is a decomposition of the total $m times m$ boundary matrix $diff$, then for any $1 <= i lt j <= m$ we have $low_R lr([j]) = i$ if and only if $r_diff (i , j) = 1$. As a result, for $1 <= i lt j <= m$, we have: 
	
	$ low_R lr([j]) = i arrow.l.r.double r_R (i , j) eq.not 0 arrow.l.r.double r_diff (i , j) eq.not 0 $ 
	
	Extending this result to @eq:lower_left_rank can be seen by observing that in the decomposition, $R = diff V$, the matrix $V$ is full-rank and obtained from the identity matrix $I$ via a sequence of rank-preserving (elementary) left-to-right column additions.
]

#proof([of @prop:mu_betti_1])[
		We first need to show that $beta_p^(i , j)$ can be expressed as a sum of rank functions. Note that by the rank-nullity theorem, so we may rewrite @eq:pbn as: 
	
	$ beta_p^(i , j) = dim (C_p (K_i))) - dim lr((B_(p - 1) (K_i))) - dim lr((Z_p (K_i) sect B_p (K_j))) $ 
	
	The dimensions of groups $C_p (K_i)$ and $B_p (K_i)$ are given directly by the ranks of diagonal and boundary matrices, yielding: 
	
	$ beta_p^(i , j) = rank lr((I_p^(1 , i))) - rank lr((diff_p^(1 , i))) - dim (Z_p (K_i) sect B_p (K_j)) $ 
	
	To express the intersection term, note that we need to find a way to express the number of $p$-cycles born at or before index $i$ that became boundaries before index $j$. Observe that the non-zero columns of $R_(p+1)$ with index at most $j$ span $B_p (K_j)$, i.e. ${med col_(R_(p+1)[k]) eq.not 0 | k in [j] med} in Im(partial_(p+1)^(,j))$. Now, since the low entries of the non-zero columns of $R_(p+1)$ are unique, we have: 
	
	$ dim(Z_p (K_i) sect B_p (K_i)) = abs(Gamma_p^(i,j)) $ <eq:s1>

	where $Gamma_p^(i,j) = {med col_(R_(p+1)[k]) eq.not 0 | 1 <= low_(R_(p+1)) [k] <= i  }$. Consider the complementary matrix $macron(Gamma)_p^(i,j)$ given by the non-zero columns of $R_(p+1)$ with index at most $j$ that are not in $Gamma_p^(i,j)$, i.e. the columns satisfying $low_(R_(p+1))[k] > i$. Combining rank-nullity with the observation above, we have: 

	$ macron(Gamma)_p^(i,j) = dim(B_p (K_j)) - abs(Gamma_p^(i,j)) = rank(R_(p+1)^(i+1,j)) $ <eq:s2>
	
	Combining equations @eq:s1 with @eq:s2 yields:

	$ dim(Z_p(K_i) sect B_p(K_j)) = abs(Gamma_p(i,j)) = dim(B_p (K_j)) - |macron(Gamma)_p^(i,j)| = rank(R_(p+1)^(1,j)) - rank(R_(p+1)^(i+1,j)) $ <eq:s3>

	Observing the final matrices in @eq:s3 are _lower-left_ submatrices of $R_{p+1}$, thus the final expression @eq:betti_four follows by applying @lemma:rank repeatedly. 
] <proof:mu_betti_1>

== Combinatorial Laplacians <sec:laplacian_theory>
The natural extension of the graph Laplacian $L$ to simplicial complexes is the #emph[$p$-th combinatorial Laplacian] $Delta_p$, whose explicit matrix representation is given by @eq:comb_lap. Indeed, when $p = 0$, $Delta_0 (K) = diff_1 diff_1^T = L$ recovers the graph Laplacian. As with boundary operators, $Delta_p (K)$ encodes simplicial homology groups in its nullspace, a result known as the discrete Hodge Theorem @lim2020hodge: 

$ tilde(H)_p (K semi bb(R)) tilde.eq ker (Delta_p (K)) , quad beta_p = nullity (Delta_p (K)) $ 

The fact that the Betti numbers of $K$ may be recovered via the nullity of $Delta_p (K)$ has been well studied (see e.g. Proposition 2.2 of @horak2013spectra). In fact, as pointed out by @horak2013spectra, one need not only consider $Delta_p$ as the spectra of $Delta_p$, $L_p^up$, and $L_p^dn$ are intrinsically related by the identities: 

$ Lambda (Delta_p (K)) ≐ Lambda lr((L_p^up)) union.dot Lambda lr((L_p^dn)), quad quad Lambda lr((L_p^up)) ≐ Lambda lr((L_(p + 1)^dn)) $ 

where $A ≐ B$ and $A union.dot B$ denotes equivalence and union between the #emph[non-zero] elements of the multisets $A$ and $B$, respectively. Moreover, all three operators $Delta_p$, $L_p^up$, and $L_p^dn$ are symmetric, positive semidefinite, and compact—thus, for the purpose of estimating $beta_p$, it suffices to consider only one family of operators.

// To translate the continuity results from @def:smooth_mu[\[def:smooth\_mu\]] to any of the Laplacian operators above, we must consider weighted versions. Here, a _weight function_ is a non-negative real-valued function defined over the set of all faces of $K$: $ w:K -> bb(R)_+ $ The set of weight functions and the choice of scalar product on $C^p (K , bb(R))$ wherein elementary cochains are orthogonal are in one-to-one correspondence #cite() (see Appendix @sec:inner_products[\[sec:inner\_products\]]). In this way, we say that the weight function _induces_ an inner product on $C^p (K , bb(R))$: $ angle.l thin f , g thin angle.r_w = sum_(sigma in K^p) w (sigma) f lr((lr([sigma]))) g lr((lr([sigma]))) $ Moreover, Laplacian operators are uniquely determined by the choice of weight function. This correspondence permits us to write the matrix representation of $Delta_p$ explicitly: $ Delta_p (K , w) eq.delta W_p^+ diff_(p + 1) W_(p + 1) diff_(p + 1)^T thin + thin diff_p^T W_p^+ diff_p W_(p + 1) $ where $W_p = diag lr((brace.l thin w (sigma_i) thin brace.r_(i = 1)^n))$ represents a non-negative diagonal matrices restricted $sigma in K^p$ and $W^+$ denotes the pseudoinverse. Note that @eq:weighted_up_laplace[\[eq:weighted\_up\_laplace\]] recovers @eq:comb_lap[\[eq:comb\_lap\]] in the case where $w$ is the constant map $w (sigma) = 1$, which we call the _unweighted_ case.

// Unfortunately, various difficulties arise with weighting combinatorial Laplacians with non-constant weight functions, such as asymmetry, scale-dependence, and spectral instability. Indeed, observe that in general neither terms in @eq:weighted_up_laplace[\[eq:weighted\_up\_laplace\]] are symmetric unless $W_p = I_n$ (for $L_p^up$) or $W_(p + 1) = I_m$ (for $L_p^dn$). However, as noted in @memoli2022persistent, $L_p^up$ may be written as follows: $ L_p^up = W_p^+ diff_(p + 1) W_(p + 1) diff_(p + 1)^T = W_p^(+ slash 2) #scale(x: 120%, y: 120%)[paren.l] W_p^(+ slash 2) diff_(p + 1) W_(p + 1) diff_(p + 1)^T W_p^(+ slash 2) #scale(x: 120%, y: 120%)[paren.r] W_p^(1 slash 2) $ Since @eq:l_up[\[eq:l\_up\]] is of the form $W^+ P W$ where $P in S_n^+$ and $W$ is a non-negative diagonal matrix, this rectifies the symmetry problem. Towards bounding the spectra of $L_p^up$, Horek and Jost #cite() propose _normalizing_ $Delta_p$ by augmenting $w$’s restriction to $K^p$: $ w (tau) = sum_(tau in diff (sigma)) w (sigma) quad forall #h(0em) tau in K^p , thin sigma in K^(p + 1) $ Substituting the weights of the $p$-simplices in this way is equivalent to mapping $W_p |-> cal(D)_p$ where $cal(D)_p$ is the _diagonal degree matrix_. The corresponding substitution in @eq:l_up[\[eq:l\_up\]] yields the _weighted combinatorial normalized Laplacian_ (up-)operator: \$\$\\begin{aligned}
// \\label{eq:normalized\_up\_lap}
//      \\mathcal{L}\_p^{\\text{up}} \= (\\mathcal{D}\_p)^{+/2} \\partial\_p W\_{p+1} \\partial\_p^T (\\mathcal{D}\_p)^{+/2} \= \\mathcal{I}\_n - \\mathcal{A}\_p^{\\text{up}} \\addtocounter{equation}{1}\\tag{\\theequation}
// \\end{aligned}\$\$ where $cal(A)_p^up$ is a weighted adjacency matrix, and $cal(I)_n$ is the identity matrix with $cal(I) (tau) = upright(s i g n) (w (tau))$ (see Section @sec:comb_lap[\[sec:comb\_lap\]]). The primary benefit of this normalization is that it guarantees $Lambda lr((cal(L)_p^up)) subset.eq lr([0 , p + 2])$ for any choice of weight function, from which one obtains several useful implications, such as tight bounds on the spectral norm #cite(). The same results holds for up-, down-, and combinatorial Laplacians. Moreover, as we will show in a subsequent section, one obtains stability properties with degree-normalization not shared otherwise.

// #strong[Remark 4]. Compared to @eq:l_up[\[eq:l\_up\]], is it worth remarking that one important quality lost in preferring $cal(L)_p^up$ over $L_p^up$ is diagonal dominance.

== Laplacian matvec <app:lap_matvec>
Given a simple undirected graph $G = (V , E)$, let $A in brace.l 0 , 1 brace.r^(n times n)$ denote its binary adjacency matrix satisfying $A lr([i , j]) = 1 arrow.l.r.double i tilde.op j$ if the vertices $v_i , v_j in V$ are adjacent in $G$, and let $D = diag lr((brace.l thin deg (v_i) thin brace.r))$ denote the diagonal _degree_ matrix, where $deg (v_i) = sum_(j eq.not i) A lr([i , j])$. The _graph Laplacian_'s adjacency, incidence, and element-wise definitions are: 

$ L = D - A = diff_1 compose diff_1^T thin , quad quad L thin lr([i , j]) = cases(delim: "{", deg (v_i) & text(" if ") i = j, - 1 & text(" if ") i tilde.op j, 0 & text(" if ") i tilde.not j) $ 

By using the adjacency relation $i tilde.op j$ as in @chung1997spectral, the linear and quadratic forms of $L$ may be succinctly expressed as:

$ L(x)_i = deg(v_i) dot x_i - sum_(i tilde.op j) x_j, quad x^T L x = sum_(i tilde.op j) (x_i - x_j)^2 $ <eq:lap_quad_form> 

If $G$ has $m$ edges and $n$ vertices taking labels in the set $lr([n])$, observe computing the product from @eq:lap_quad_form requires just $O (m)$ time and $O (n)$ storage via two edge traversals: one to accumulate vertex degrees and one to remove components from incident edges. By precomputing the degrees, the operation reduces further to a single $O (n)$ product and $O (m)$ edge pass, which is useful when repeated evaluations for varying values of $x$ are necessary.

To extend the two-pass algorithm outlined above for $p gt 0$, we first require a generalization of the connected relation $i tilde.op j$ from @eq:lap_quad_form. Denote with $co (tau) = brace.l thin sigma in K^(p + 1) divides tau subset sigma thin brace.r$ the set of proper cofaces of $tau in K^p$, or _cofacets_, and the (weighted) _degree_ of $tau in K^p$ with: $ deg_w (tau) = sum_(sigma in co (tau)) w (sigma) $ Note setting $w (sigma) = 1$ for all $sigma in K$ recovers the integral notion of degree representing the number of cofacets a given $p$-simplex has. Now, since $K$ is a simplicial complex, if the faces $tau , tau prime$ share a common cofacet $sigma in K^(p + 1)$, this cofacet $sigma$ is in fact _unique_ in the sense that $brace.l sigma brace.r = co (tau) sect co (tau prime)$ (see @goldberg2002combinatorial for more details). Thus, we may use a relation $tau tilde.op^sigma tau prime$ to rewrite $L_p^up$ element-wise: 

$ L_p^up (tau , tau prime) = cases(delim: "{", deg_w (tau) dot.op w^+ (tau) & text(" if ") tau = tau prime, s_(tau , tau prime) dot.op w^(+ slash 2) (tau) dot.op w (sigma) dot.op w^(+ slash 2) (tau prime) & text(" if ") tau tilde.op^sigma tau prime, 0 & text(" otherwise")) $ 

where $s_(tau , tau prime) = sgn lr((lr([tau]) , diff lr([sigma]))) thin dot.op thin sgn lr((lr([tau]) , diff lr([sigma])))$. Ordering the $p$-faces $tau in K^p$ along a total order and choosing an indexing function $h:K^p -> lr([n])$ enables explicit computation of the corresponding matrix-vector product: 

$ lr((L_p^up thin x))_i = deg_w (tau_i) dot.op w^+ (tau_i) dot.op x_i + w^(+ slash 2) (tau_i) sum_(tau_j tilde.op^sigma tau_i) s_(tau_i , tau_j) dot.op x_j dot.op w (sigma) dot.op w^(+ slash 2) (tau_j) $ <eq:l_up_matvec>

Observe $v -> L_p^up v$ can be evaluated now via a very similar two-pass algorithm as described for the graph Laplacian by simply enumerating the boundary chains of the simplices of $K^(p + 1)$. 

Below is pseudocode showing how to evaluate a weighted (up) Laplacian matrix-vector multiplication built from a simplicial complex $K$ with $m = abs(K^(p + 1))$ and $n = abs(K^p)$ in $O (m)$ time when $m gt n$, assuming $p$ is considered a small constant. 
Key to the runtime of the linear runtime operation is the constant-time determination of orientation between $p$-faces ($s_(tau , tau prime)$) and—for sparse complexes—the use of a deterministic $O (1)$ hash table $h:K^p -> lr([n])$ for efficiently determining the appropriate input/output offsets ($i$ and $j$). 

#figure(
	kind: "algorithm",
	supplement: [Algorithm],
	caption: [Matrix-free up-Laplacian matrix-vector multiplication algorithm.],
	pseudocode-list(booktabs: true, line-gap: 0.50em)[
		#h(-0.20cm) *Require:* Fixed oriented complex $K$ of size $N = abs(K)$, $n = abs(K^p)$, $m = abs(K^(p+1))$ \
		#h(-0.38cm) *Optional:* Weight arrays $w_(p+1) in bb(R)_+^m$ and $w_p in bb(R)_+^n$ \
		#h(-0.38cm) *Output:* $y = (W_p compose partial_(p+1) compose W_(p+1) compose partial_(p+1)^T compose W_p) x$ \
		#v(-1.75em)
		#h(-0.38cm) #align(center, line(length: 100%, stroke: 0.5pt))
		+ Construct hash function $h: K^p -> [n]$
		+ $deg_w <- 0$
		+ for $sigma in K^(p+1)$, $k in [m]$ do: 
			+ for $tau in partial[sigma]$ do:
				+ $deg_w [h(tau)] <- deg_w [h(tau)] + w_p [h(tau)] dot w_(p+1) [k] dot w_p [h(tau)]$ 
		+ *function* UpLaplacianMatvec($x in bb(R)^n$)
			+ $y <- deg_w dot.circle x$ (element-wise product)
			+ for $sigma in K^(p+1), k in [m]$ do: 
				+ for $tau,tau' in partial[sigma] times partial[sigma]$ where $tau eq.not tau'$ do: 
					+ $s_(tau, tau') <- sgn([tau], partial[sigma]) dot sgn([tau'], partial[sigma])$
					+ $i, j <- h(tau), h(tau')$
					+ $y_i <- y_i + s_(tau, tau') dot x_j dot w_p[i] dot w_(p+1)[k] dot w_p[j]$ 
			+ return $y$ 
	]
) <alg:lap_matvec>


// #figure([#image("../images/laplacian_matvec_alg.png", width: 95%)],
//   caption: [
//     Combinatorial Laplacian matrix-vector multiplication.
//   ]
// ) <alg:lap_matvec>

In general, the signs of the coefficients $sgn lr((lr([tau]) , diff lr([sigma])))$ and $sgn lr((lr([tau prime]) , diff lr([sigma])))$ depend on the position of $tau , tau prime$ as summands in $diff lr([sigma])$, which itself depends on the orientation of $lr([sigma])$. Thus, evaluation of these sign terms takes $O (p)$ time to determine for a given $tau in diff lr([sigma])$ with $dim (sigma) = p$, which if done naively via line (12) in the pseudocode @alg:lap_matvec increases the complexity of the algorithm. However, observe that the sign of their product is in fact invariant in the orientation of $lr([sigma])$ (see Remark 3.2.1 of @goldberg2002combinatorial)—thus, if we fix the orientation of the simplices of $K^p$, the sign pattern $s_(tau , tau prime)$ for every $tau tilde.op^sigma tau prime$ can be precomputed and stored ahead of time, reducing the evaluation $s_(tau , tau prime)$ to $O (1)$ time and $O (m)$ storage. Alternatively, if the labels of the $p + 1$ simplices $sigma in K^(p + 1)$ are given an orientation induced from the total order on $V$ and $p$ is a small constant, we can remove the storage requirement entirely and simply fix the sign pattern during the computation.

A subtle but important aspect of algorithmically evaluating @eq:l_up_matvec is the choice of indexing function $h:K^p -> lr([n])$. This map is necessary to deduce the contributions of the components $x_ast$ during the operation (line (13)). While this task may seem trivial as one may use any standard associative array to generate this map, typical implementations that rely on collision-resolution schemes such as open addressing or chaining only have $O (1)$ lookup time in expectation. Moreover, empirical testing suggests that line (13) in @alg:lap_matvec can easily bottleneck the entire computation due to the scattered memory access such collision-resolution schemes may involve. One solution avoiding these collision resolution schemes that exploits the fact that $K$ is fixed is to build an order-preserving _perfect minimal hash function_ (PMHF) $h:K^p -> lr([n])$. It is known how to build PMHFs over fixed input sets of size $n$ in $O (n)$ time and $O (n log m)$ bits with deterministic $O (1)$ access time @botelho2005practical. Note that this process happens only once for a fixed simplicial complex $K$: once $h$ has been constructed, it is fixed for every $mono(m a t v e c)$ operation.


== Choosing a weight function 
// TODO: don't know I should include this
Given @eq:inner_product_cochain, a natural question to ask whether there exists a weight function $w$ that is more appropriate for rank computations. Since different weight choices yield different eigenvalue distribution in $cal(L)_p$, the ideal choice is one that is the most amenable to compute. 
For the purpose of improving the condition number of the underlying Laplacian, consider the following optimization problem: 

$ max_(w in bb(R)^n) & lambda_min (cal(L)_p (w)) \ 
	text("subject to") & w > 0, med bold(1)^T w  = 1 
$ <eq:alg_connectivity>

In the spectral graph theory, a specialization of this problem (for $p = 1$) arises under the guise of related problems, such as maximizing _algebraic connectivity_ under a fixed total edge weight or finding the "fastest mixing time" of a Markov process. 
Fortunately, @eq:alg_connectivity is a convex optimization problem, which can be formulated as a semi-definite program (SDP) with variables $gamma, beta in bb(R), w in bb(R)^n$ under positivity and sum-to-one constraints:

$
max_(gamma in bb(R)) med & gamma \
text("subject to") med & gamma I prec.eq partial_1 D_1 (w) partial_1^T + beta bold(1)bold(1)^T, quad w > 0, quad bold(1)^T w = 1 
$

In general, we do not know if this approach generalizes for higher-order Laplacians, but in practice we found the weight function $w^ast$ that optimizes @eq:alg_connectivity to indeed be the easiest to approximate. 

// Despite the success of iterative methods in efficiently solving linear systems manifesting from diagonally dominant sparse matrices is #cite(), such advancements have not yet been extended to the persistence setting.

// TODO
// === Output sensitive multiplicity and Betti <output-sensitive-multiplicity-and-betti>
// We record this fact formally with two corollaries. Let $upright(R)_p (k)$ denotes the complexity of computing the rank of square $k times k$ matrix with at most $O (lr((p + 1)) k)$ non-zero $bb(F)$ entries. Then we have:

// #strong[Corollary 4]. #emph[Given a filtration $K_bullet$ of size $N = abs(K_bullet)$ and indices $(thin i , j thin) in Delta_+^N$, computing $beta_p^(i , j)$ using expression @eq:betti_four[\[eq:betti\_four\]] requires $O #scale(x: 120%, y: 120%)[paren.l] max brace.l upright(R)_p (n_i) , upright(R)_(p + 1) (m_j) brace.r #scale(x: 120%, y: 120%)[paren.r]$ time, where $n_i = abs(K_i^p)$ and $m_j = abs(K_j^(p + 1))$.]

// Observe the relation \$\\partial\_{p%
//     \\raisebox{0.18ex}{\\scaleobj{0.55}{+}}
// %  \\raisebox{\\dimexpr(\\fontcharht\\font\`X-\\height+\\depth)/2\\relax}{\\scaleobj{0.5}{+}}%
// 1}^{i %
//     \\raisebox{0.18ex}{\\scaleobj{0.55}{+}}
// %  \\raisebox{\\dimexpr(\\fontcharht\\font\`X-\\height+\\depth)/2\\relax}{\\scaleobj{0.5}{+}}%
//  1, j} \\subseteq \\partial\_{p%
//     \\raisebox{0.18ex}{\\scaleobj{0.55}{+}}
// %  \\raisebox{\\dimexpr(\\fontcharht\\font\`X-\\height+\\depth)/2\\relax}{\\scaleobj{0.5}{+}}%
// 1}^{1, j}\$ 
// implies the dominant cost of computing @eq:betti_four[\[eq:betti\_four\]] lies in computing either $rank lr((diff_p^(1 , i)))$ or $rank lr((diff_(p + 1)^(1 , j)))$, which depends on the relative sizes of $abs(K^p)$ and $abs(K^(p + 1))$. In contrast, $mu_p^R$ is localized to the pair $(K_i , K_l)$ and depends only on the $(p + 1)$-simplices in the interval $lr([i , l])$, yielding the following corollary.

// #strong[Corollary 5]. #emph[Given a filtration $K_bullet$ of size $N = abs(K_bullet)$ and a rectangle $R = lr([i , j]) times lr([k , l])$ with indices $0 <= i lt j <= k lt l <= N$, computing $mu_p^R$ using expression @eq:mu_four[\[eq:mu\_four\]] requires $O (upright(R)_(p + 1) lr((m_(i l))))$ time $m_(i l) = abs(K_l^(p + 1) )- abs(K_i^(p + 1))$.]




// === Proofs <proofs>


// #proof([of @lemma:rank])[
// 	The Pairing Uniqueness Lemma @dey2022computational asserts that if $R = diff V$ is a decomposition of the total $m times m$ boundary matrix $diff$, then for any $1 <= i lt j <= m$ we have $low_R lr([j]) = i$ if and only if $r_diff (i , j) = 1$. As a result, for $1 <= i lt j <= m$, we have: 
	
// 	$ low_R lr([j]) = i arrow.l.r.double r_R (i , j) eq.not 0 arrow.l.r.double r_diff (i , j) eq.not 0 $ 
	
// 	Extending this result to @eq:lower_left_rank can be seen by observing that in the decomposition, $R = diff V$, the matrix $V$ is full-rank and obtained from the identity matrix $I$ via a sequence of rank-preserving (elementary) left-to-right column additions.
// ] <proof-of-lemma-1>

// #proof([of @prop:mu_betti_1])[ 
// 	We first need to show that $beta_p^(i , j)$ can be expressed as a sum of rank functions. Note that by the rank-nullity theorem, so we may rewrite @eq:pbn as: 
	
// 	$ beta_p^(i , j) = dim (C_p (K_i))) - dim lr((B_(p - 1) (K_i))) - dim lr((Z_p (K_i) sect B_p (K_j))) $ 
	
// 	The dimensions of groups $C_p (K_i)$ and $B_p (K_i)$ are given directly by the ranks of diagonal and boundary matrices, yielding: 
	
// 	$ beta_p^(i , j) = rank lr((I_p^(1 , i))) - rank lr((diff_p^(1 , i))) - dim (Z_p (K_i) sect B_p (K_j)) $ 
	
// 	To express the intersection term, note that we need to find a way to express the number of $p$-cycles born at or before index $i$ that became boundaries before index $j$. Observe that the non-zero columns of $R_(p+1)$ with index at most $j$ span $B_p (K_j)$, i.e. ${med col_(R_(p+1)[k]) eq.not 0 | k in [j] med} in Im(partial_(p+1)^(,j))$. Now, since the low entries of the non-zero columns of $R_(p+1)$ are unique, we have: 
	
// 	$ dim(Z_p (K_i) sect B_p (K_i)) = abs(Gamma_p^(i,j)) $ <eq:s1>

// 	where $Gamma_p^(i,j) = {med col_(R_(p+1)[k]) eq.not 0 | 1 <= low_(R_(p+1)) [k] <= i  }$. Consider the complementary matrix $macron(Gamma)_p^(i,j)$ given by the non-zero columns of $R_(p+1)$ with index at most $j$ that are not in $Gamma_p^(i,j)$, i.e. the columns satisfying $low_(R_(p+1))[k] > i$. Combining rank-nullity with the observation above, we have: 

// 	$ macron(Gamma)_p^(i,j) = dim(B_p (K_j)) - abs(Gamma_p^(i,j)) = rank(R_(p+1)^(i+1,j)) $ <eq:s2>
	
// 	Combining equations @eq:s1 with @eq:s2 yields:

// 	$ dim(Z_p(K_i) sect B_p(K_j)) = abs(Gamma_p(i,j)) = dim(B_p (K_j)) - |macron(Gamma)_p^(i,j)| = rank(R_(p+1)^(1,j)) - rank(R_(p+1)^(i+1,j)) $ <eq:s3>

// 	Observing the final matrices in @eq:s3 are _lower-left_ submatrices of $R_{p+1}$, thus the final expression @eq:betti_four follows by applying @lemma:rank repeatedly. 
// ] <proof-of-proposition-1>


// #proof[
// 	The above result immediately follows by applying the fact that $lim_(tau -> 0^+) norm(Phi_tau (X))_ast = rank (X)$ to each of the constitutive terms of $hat(mu)_(p , tau)^R$ and $hat(beta)_(p , tau)^(i , j)$.
// ] <proofs-of-basic-properties>