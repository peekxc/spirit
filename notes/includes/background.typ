#import "../math_ops.typ": * 
#import "../theorems.typ": *

== Technical background <sec:betti_derivation>
The following results summarize some technical observations motivating this effort, which will be used in several proofs. Though these observations serve as background material, they contextualize our non-traditional computation of the rank invariant (@cor:rank_reduction) and serve as the motivation for this work.

Among the most widely known results for persistence is the structure theorem @zomorodian2004computing, which shows 1-parameter persistence modules decompose in an _essentially unique_ way. Computationally, the corresponding Pairing Uniqueness Lemma @cohen2006vines asserts that if $R = diff V$ decomposes the boundary matrix $diff in bb(F)^(N times N)$ to a _reduced_ matrix $R in bb(F)^(N times N)$ using left-to-right column operations, then: 

$ R lr([i , j]) eq.not 0  arrow.l.r.double  rank lr((diff^(i , j))) - rank lr((diff^(i mono("+") 1 , j))) + rank lr((diff^(i mono("+") 1 , j upright("-") 1))) - rank lr((diff^(i , j upright("-") 1))) eq.not 0 $ <eq:uniq_pivot>

where $diff^(i , j)$ denotes the lower-left submatrix defined by the first $j$ columns and the last $m - i + 1$ rows (rows $i$ through $m$, inclusive). Thus, the existence of non-zero "pivot" entries in $R$ may be inferred entirely from the ranks of certain submatrices of $diff$. Part of the validity of @eq:uniq_pivot can be attributed to the following Lemma:

#lemma[
	Given filtration $(K , f)$ of size $N = abs(K)$, let $R = diff V$ denote the decomposition of the filtered boundary matrix $diff in bb(F)^(N times N)$. Then, for any pair $(i , j)$ satisfying $1 <= i lt j <= N$, we have: 

	$ rank lr((R^(i , j))) = rank lr((diff^(i , j))) $ <eq:lower_left_rank>

	Equivalently, all lower-left submatrices of $diff$ have the same rank as their corresponding submatrices in $R$.
] <lemma:rank>

An explicit proof of both of these facts can be found in @dey2022computational, though the latter was also noted in passing by Edelsbrunner @edelsbrunner2000topological. 
Though typically viewed as minor facts needed to prove the correctness of the reduction algorithm, the implications of these two observations are are quite general, as recently noted by @bauer2022keeping:

#corollary("Bauer et al. " + [@bauer2022keeping])[
	Any persistence algorithm which preserves the ranks of the submatrices $diff^(i , j) (K , f)$ for all $i , j in lr([N])$ satisfying $1 <= i lt j <= N$ is a valid persistence algorithm.
] <cor:valid_pers>

Indeed, though $R$ is not unique, its non-zero pivots are, and these pivots _define_ the persistence diagram. Moreover, due to @eq:lower_left_rank, both $beta_p^ast$ and $mu_p^ast$ may be written as a sum of ranks of submatrices of $diff_p$ and $diff_(p + 1)$:

#corollary[
	Given a fixed $p >= 0$, a filtration $(K , f)$ with filtration values $brace.l thin a_i thin brace.r_(i = 1)^N$, and a rectangle $R = lr([a_i , a_j]) times lr([a_k , a_l]) subset Delta_+$, the persistent Betti and multiplicity functions may be written as:
	$ beta_p^(a_i, a_j) (K, f) = rank(C_p (K_i)) - rank(partial_p^(1,i)) - rank(partial_(p+1)^(1,j)) - rank(partial_(p+1)^(i+1,j)) $ <eq:betti_four>

	$ mu_p^R (K, f) = rank(partial_(p+1)^(j+1, k)) - rank(partial_(p+1)^(i+1, k)) - rank(partial_(p+1)^(j+1, l)) + rank(partial_(p+1)^(i+1, l)) $ <eq:mu_four>

] <cor:rank_reduction>

Though @eq:mu_four was pointed out by Cohen-Steiner et al. in @cohen2006vines and exploited computationally by Chen & Kerber in @chen2011output, to the authors knowledge the only explicit derivation and proof of @eq:betti_four is given by Dey & Wang @dey2022computational. For completeness, we give our own detailed proof of @cor:rank_reduction in @sec:proofs. In practice, neither expressions seem used or even implemented in any commonly used persistence software.

Two important properties of the expressions from @cor:rank_reduction are: (1) they are comprised strictly of _rank_ computations, and (2) all terms involve _unfactored_ boundary matrices. Coupled with measure-theoretic perspectives on persistence @chazal2016structure, the former suggests variational perspectives of the rank function might yield interesting spectral relaxations of @eq:betti_four and @eq:mu_four useful for e.g. optimization purposes. Both the latter property with @cor:valid_pers suggests a potentially new means of computing persistence information _without_ matrix reduction. 
Moreover, combining these observations suggests advances made in other areas of applied mathematics may be readily exploited, such as the rich theory of matrix functions @bhatia2013matrix, the tools developed as part of "The Laplacian Paradigm" @teng2010laplacian, or the recent connections between rank and trace estimation @ubaru2016fast. The rest of the paper is dedicated to exploring these connections and their implications.
