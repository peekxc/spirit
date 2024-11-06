#import "../math_ops.typ": * 
#import "../theorems.typ": *

= Applications & Experiments <sec:applications>
== Filtration optimization <filtration-optimization>
It is common in TDA for the filter function $f:K -> bb(R)$ to depend on hyper-parameters. For example, prior to employing persistence, one often removes outliers from point set $X subset bb(R)^d$ via some density-based pruning heuristic that itself is parameterized. This is typically necessary due to the fact that, though stable under Hausdorff noise @cohen2005stability, diagrams are notably unstable against _strong outliers_—even one point can ruin the summary. As an exemplary use-case of our spectral-based method, we re-cast the problem of identifying strong outliers below as a problem of _filtration optimization_.

#figure([#image("../images/codensity_ex.png", width: 100%)],
	placement: top, 
  caption: [
    From left to right: Delaunay complex $K$ realized from point set $X subset bb(R)^2$ sampled with multiple types of noise around $S^1$ (colored by codensity at optimal $alpha^ast approx 1 slash 2$); codensity vineyard of $(K , f_alpha)$ across varying bandwidths $alpha$ and a fixed point $(a , b) in Delta_+$; Tikhonov relaxations $hat(beta)_p^(a , b) (alpha)$ at varying regularization ($tau$) and sign width ($omega$) values. 
  ]
)
<fig:codensity_opt>

Consider a Delaunay complex $K$ realized from a point set $X subset bb(R)^2$ sampled around $S^1$ affected by both Hausdorff noise and strong outliers, shown in @fig:codensity_opt. One approach to detect the presence of $S^1$ in the presence of such outliers is maximize $beta_p^(a , b) (alpha)$ for some appropriately chosen $(a , b) in Delta_+$ over the pair $(K , f_alpha)$, where $f_alpha:X -> bb(R)_+$ is a kernel (co)-density estimate: 

$ alpha^ast = argmax_(alpha in bb(R)) beta_p^(a , b) (K , f_alpha), quad text(" where ") f_alpha (x) = frac(1, n alpha) sum_i C (cal(K))) - cal(K)_alpha lr((x_i - x)) $ <eq:betti_opt>

where $C (cal(K)_alpha)$ is a normalizing constant that depends on the choice of kernel, $cal(K)_alpha$. Intuitively, if there exists a choice of bandwidth $alpha^ast$ which distinguishes strong outliers from Hausdorff noise clustered around $S^1$, then that choice of bandwidth should exhibit a highly persistent pair $lr((a^ast , b^ast)) in dgm_1 lr((K , f_(alpha^ast)))$. Thus, if we place a corner point $(a , b)$ satisfying $a^ast <= a$ and $b lt b^ast$ for some $alpha gt 0$, we expect $beta_p^(a , b) (alpha) = 1$ to near the optimal bandwidth $alpha^ast$—matching the first Betti number of $S^1$—and $0$ otherwise.

In @fig:codensity_opt, we depict the dimension-$1$ vineyard of a simple Delaunay complex and codensity pair $(K , f_alpha)$, along with the sieve point $(a , b)$ and the region wherein $S^1$ is accurately captured by persistence. As $beta_p^(a , b)$ is an integer-valued invariant, it is discontinuous and difficult to optimize; in contrast, we know from @prop:operator_props that we can obtain a continuous and differentiable relaxation of $beta_p^(a , b)$ by replacing $beta_p^(a , b) |-> hat(beta)_p^(a , b)$ in @eq:betti_opt, enabling the use of first-order optimization techniques. By using the Tikhonov regularization from @eq:tikhonov_1, we obtain continuously varying objective curves from $hat(beta)_p^(a , b) (alpha semi tau)$ which are guaranteed to have the same maxima as $beta_p^(a , b) (alpha)$ as $tau -> 0$, as shown in @fig:codensity_opt. Observe lower values of $tau$ lead to approximations closer to the rank (black) at the cost of smoothness, while larger values can yield very smooth albeit possibly uninformative relaxations. Practical optimization of these types of objective surfaces can be handled via _iterative thresholding_, a technique which alternates between gradient steps to reduce the objective and thresholding steps to enforce the rank constraints. We leave the tuning of such optimizers to future work.

== Topology-guided simplification <topology-guided-simplification>
In many 3D computer graphics applications, one would like to simplify a given simplicial or polygonal mesh embedded in $bb(R)^3$ so as to decrease its level of detail (LOD) while retaining its principal geometric structure(s). Such simplifications are often necessary to improve the efficiency of compute-intensive tasks that depend on the size of the mesh (e.g. rendering). Though many simplification methods developed to preserve geometric criteria (e.g. curvature, co-planarity) are now well known (see @heckbert1997survey for an overview), _topology-preserving_ simplification techniques are relatively sparse, especially for higher embedding dimensions. Moreover, such procedures typically restrict to operations that preserve _local_ notions of topology, such as the genus of a feature's immediate neighborhood or the property of being a manifold. These operations are known to greatly limit the amount of detail decimation algorithms can remove.

As a prototypical application of our proposed relaxation, we re-visit the mesh simplification problem under #emph[persistence-based] constraints. In contrast to @fugacci2020topology, we forgo the use of persistence-preserving operations and instead opt for a simpler strategy: we perform an exponential search on a given sequence of simplifications, settling on the largest simplification found.

#figure([#image("../images/elephant_sparsify.png", width: 80%)],
  caption: [
    (Top) Meshes filtered and colored by eccentricity at varying levels of simplification; (middle) their diagrams and topological constraints; (bottom) simplification thresholds tested by an exponential search, on a logarithmic scale. The color/shape of the markers indicate whether the corresponding meshes meet (green triangle) or do not meet (red x) the topological constraints of the sieve—the gray marker identifies the original mesh (not used in the search). Black dashed circles correspond with the meshes in the top row. 
  ]
)<fig:elephant_sparsify>

We show an exemplary application of this idea in @fig:elephant_sparsify in sparsifying a mesh of an elephant. To filter the mesh in a geometrically meaningful way, we use the (geodesic) _eccentricity_ function, which assigns points $x in X$ in a metric space $(X, d_X)$ a non-negative value representing the distance that point is from the center of the mesh: 

$ E_p (x) = lr(((sum_(x' in X) d_X (x, x')^p) / N))^(1/p) $ <eq:eccentricity>

We may extend $E_p (x)$ to simplices $sigma in K$ by taking the max $f(sigma) = max_(v in sigma) d_X (x, v)$. Note this does not require identifying a center point in the mesh. Intuitively, the highly persistent $H_1$ classes in mesh carrying the most detail corresponds to "tubular" features that persist from the center; the four legs, the trunk, the two ears, and two tusks.
In this example, four rectangular-constraints are given which ensure the simplified elephants retain these features with certain minimal persistence $delta gt 0$.

Note that neither persistence diagrams nor deformation-compatible simplicial maps were needed to perform the sparsification, just a priori knowledge of which areas of $Delta_+$ to check for the existence of persistent topological features. We defer a full comparison of the sparsification application as future work.  

== Topological time series analysis <sec:topological_time_series>

// In many time series applications, detecting periodic behavior can prove a difficult yet useful thing to estimate. For example, in medical images contexts, it is necessary to preprocess the data to remove noise and outliers that otherwise obscure the presence of recurrent behavior. 
// method understands periodicity as repetition ofpatterns, whatever these may be, and quantifies this reoccurrence as the degree of circularity/roundness in the generated point-cloud.

The canonical lens by which time series analysis is performed through harmonic analysis. Though mature and powerful, it can at times be more illuminating to study the data in more geometric settings. A classical example of this is Takens delay embedding theorem, which provides sufficiency conditions under which a smooth attractor of $d$-dimensional manifold can be reconstructed from its dynamics $f : M -> M$. In this context, the "reconstruction" is a phase space embedding of a given time series topologically equivalent to the original space, constructed through a _time-delay embedding_ of $f$: 

$ SW_(M, tau) f(t) = mat(delim: "[", f(t), f(t + tau), dots.h.c, f(t + M tau))^top $

where $M+1$ defines the _embedding dimension_ into $bb(R)^(M + 1)$ and $M tau$ is the _window size_ with respect to some $tau > 0$. Choosing different values of $t$ yields a collection of points called the sliding window point cloud of $f$. Under certain conditions, Takens theorem shows this reconstruction indeed has topological equivalence using the Whitney Embedding Theorem, motivating the study of the topology of $SW_(M, tau) f$ itself.
This methodology has been used to characterize non-linearity or chaotic behavior in ECG-EKG and MEG data.

The time-delay embedding requires fixing two parameters: the dimension $M in bb(N)$ and delay $tau in bb(R)_+$. Takens theorem guarantees topological equivalence if $M$ is more twice as large as $d$, thus choosing dimensions larger than $2d+1$ poses no problems beyond sampling density and computational concerns. In contrast, once $M$ is fixed, the embedding delay $tau$ strongly affects the shape of the corresponding embedding $SW_(M, tau)$. Indeed, for general functions, the geometry of the curve $t |-> SW_(M, tau) f(t)$ can be quite complicated. However, the analysis by @perea2015sliding shows that if $f$ is periodic, the roundness#footnote[Here, _roundness_ is defined as the largest radius of a ball in $bb(R)^(M + 1)$ such that the curve $SW_(M, tau)$ is tangent to at least two points from its equator.] of the sliding window point cloud is maximized as the window-size approaches the length of the period---this periodicity can be recovered by maximizing the maximum 1-dimensional persistence of the sliding window Vietoris-Rips filtration:

$ tau^ast = argmax_(tau med in med bb(R)_+)  ( max_( (a,b) med in med cal(D)_(tau, f)) abs(b - a) ), quad cal(D)_(tau, f) = dgm_1(Rips(SW_(M, tau) f)) $ <eq:optimize_tau>

For more details on this connection, see Theorem 4.5 of @perea2015sliding. 
On the positive side, the stability of persistence alongside the Whitney embedding theorem suggests that 
@eq:optimize_tau is a valid proxy objective to for the purpose of detecting periodic behavior in complex or noisy signals which might otherwise be difficult to detect using e.g. autocorrelation. On the negative side, solving @eq:optimize_tau directly presents a computational complexity issue, due to the large size of $Rips$ and the high asymptotic complexity of persistence. 

#figure(
	[#image("../images/sw1pers_mu.png", width: 100%)],
	placement: top, 
  caption: [
    Maximum SW 1-persistence for $cal(D)_(tau, f)$ at varying $tau in [2 pi c slash 8, 2 pi c slash 4]$ with $c = 7 slash 8$ alongside the Tikhonov regularized spectral multiplicity $hat(mu)_p^R (K, f_alpha)$ for regularization $epsilon = 0.10$ and box $R = [1.1, 1.2] times [3.6, 3.8]$ (left); two vineyard plots, one showing the region that determines maximum persistence (middle-top), the bottom showing the chosen box $R$ (middle-bottom); time series $f(t)$ with window-lengths $M tau$ highlighted for three values of $tau$ and their embedding plotted with PCA (right). 
  ]
)<fig:sw1pers_spirit>

//  e.g. a more typical approach to estimating $tau$, such as with the autocorrelation function. 
//  while $M$ is typically estimated using false nearest neighbors
To illustrate another application of our spectral relaxation, we consider an proxy optimization problem that is both easier and more efficient to optimize than @eq:optimize_tau. 
We begin with a simple periodic function $f$ given by: 

$ f(t) = cos(t) + cos(3 t), quad t in [0, 12 pi] $

Regarding the parameterization, we consider optimizing $tau$ over the interval $[2 pi c slash 8, 2 pi c slash 4]$ using the Tikhonov regularized spectral multiplicity $hat(mu)_p^R$ with a (large) regularization from @eq:tikhonov_1 with $epsilon = 0.10$.
To determine the proxy objective function $hat(mu)_1^R$, we fix a rectangle $R = [1.1, 1.2] times [3.6, 3.8] in Delta_+$, which we determined contains the correct maximizer by inspection of the persistence vineyard#footnote[In practice, the full vineyard wouldn't be available, and the box $R$ would need to be determined using apriori knowledge or by probing random subsets of $R subset Delta_+$ for non-trivial persistence.]. 
All other parameters determined, we plot both the (true) maximum persistence and the optimization surface of $hat(mu)_1^R$ on the top and bottom left subplot of @fig:sw1pers_spirit, respectively.  

Analogous to the Takens and Nyquist-Shannon theorems, @perea2015sliding show that for trigonometric polynomials, one loses no information if the embedding dimension is greater than twice the maximum frequency, thus we set $M = 7$ and sample $n = 300 >> 2 (3 dot 6)$ points to ensure no issues related dimension or sampling density occur. Then, we measure the maximum dimension-1 persistence of $cal(D)_(tau, f)$ across 900 equispaced delay values $tau in [0, 12 pi]$, which we plot on the top left of @fig:sw1pers_spirit. 
We observed the maximum persistence was attained at around $tau approx 0.913$; the theory from @perea2015sliding suggests that if $f$ is $L$-periodic, the delay value $tau$ which achieves the maximum persistence should be around $tau = 2 pi c  slash L approx 0.916$ where $c = M slash (M + 1)$ is the proportionality constant. 

// Since $L$ is typically unknown, we would like a more empirical way of determining $tau$ that avoids the entire persistence computation. 



// (2*np.pi / (6)) * (7/8)

// over the domain $omega = [0, 12 pi]$. Since $f$ has 6 periods in this domain, we may take a dimension 
// w = bounds[1] * d / (L * (d + 1))


// We now formalize this process. Assume the set of topological constraints are given as input via a set of pairs $brace.l thin (R_1 , c_1)) , lr((R_2 , c_2) , dots.h , (R_h , c_h) thin brace.r$, where each $R_i subset Delta_+$ prescribing rectangular areas of wherein a multiplicity constraint on the persistence is imposed. We seek to find the minimal size mesh $K$

// $ min_(alpha in bb(R)) quad & hash thin a r d) lr((K_alpha^p))\
// upright("s.t.") quad & mu_p^(R_i) (K_alpha , f_alpha) = c_i , quad forall #h(0em) R_i in cal(R) $ In other words, the set
// 
// 
// 
// 
// 6

== Low memory persistence computations <sec:low_memory>

One particular limitation of the persistence computation is its space usage. Though the initial boundary matrices (when explicitly constructed) are known to be sparse, both the constitutive matrices storing the reduced boundary chains ($R$) and the cycle representatives ($V$) are known to lose their sparsity throughout the duration of the reduction. As the number of non-zeros in $R$ affects the performance of reduction, analyzing the worst and average-case amount of "fill-in" is a subject of recent research @bauer2022keeping. Though the average case complexity is far better than the worst case, space usage is a well known to be one of the barriers antagonizing the scalability of the persistence computation. 
As the persistence diagram may be entirely determined by rank computations, we may extend the time and space complexity results from @prop:spectral_rank_complexity directly to the computation of persistence diagrams via Chen and Kerbers divide-and-conquer approach @chen2013output. 

#figure([
	#image("../images/ripser_vs_laplacian.png", width: 65%)],
	placement: top, 
  caption: [
		The "high watermark" maximum (heap) memory allocated by Ripser vs. Lanczos run with a constant degree on the Rips filtrations combinatorial Laplacian operator. To 
  ]
)
<fig:ripser_vs_laplacian>

To demonstrate the space efficiency of the matrix-free approach, we revisit the persistence computation in the context of the homology inference problem, one of the original use-cases of persistence @perea2018brief. To measure the scalability of the Lanczos method, we measure its space usage via the persistence computation on an increasingly dense point cloud sampled from the uniform measure on the 2-torus $bb(T)$ (via @diaconis2013sampling):
$ 
f(theta, phi) = ((R + r cos(theta)) cos(phi), (R + r cos(theta)) sin(phi), r sin(theta))
$ 

To ensure uniform coverage of $bb(T)$, we evaluate the Rips 1-persistence computation on landmarks via $k$-prefixes of the _greedy permutation_ of a sufficiently dense point sample, for $k$ varying along evenly spaced points from $50$ to $2500$ (yielding complexes with sizes ranging from $2.0 dot 10^4$ to $2.6 dot 10^9$ simplices, respectively).
For each $k$, we compute the Rips persistence up to the enclosing radius to ensure all pairs with finite persistence are constructed. 
The result are summarized in @fig:ripser_vs_laplacian compared to the popular software _Ripser_ @bauer2021ripser.
For an even comparison to _Ripser_, we record the maximum (heap) memory usage of both _Ripser_ and the constant-degree Lanczos method space usage by monitoring all system allocations using the software _Memray_#footnote[See https://bloomberg.github.io/memray/memory.html for an overview of how memory is measured.].
To isolate the simplices which contribute to the space usage, like _Ripser_, we use the _apparent pairs_ optimization throughout the computation to discard zero-persistence pairs (see @sec:apparent-pairs-optimization).

It's worth noting the results in @fig:ripser_vs_laplacian carry certain limitations. In particular, we only report the memory usage of both methods, rather than the time usage, as the method by Chen and Kerber is too complicated to implement. Additionally, _Ripser_ is computing persistence here with respect to $bb(Z) slash 2$ coefficients, whereas we rely on IEEE 64-bit floating precision arithmetic as a proxy for $bb(R)$ (thus the persistence diagrams are not identical). For more limitations, see @sec:concluding-remarks.
// On the positive, though _Ripser_ is very memory efficient relative to other reduction implementations, as demonstrated by @fig:ripser_vs_laplacian, the space asymptotics eventually catch up and render.  
// Since we only use a linear amount of space to perform the computations, 
// which is known to be very memory efficient relative to other reduction implementations. 


== Shape comparison via featurization <sec:shape_comparison>

Analogous to its historical use in comparing shapes (i.e. manifolds, curves, etc.) through size functions, persistence has increasingly found use in machine learning pipelines through the use of mappings from diagrams to function spaces (e.g. Hilbert spaces).
// enabling the extraction of topologically-sensitive featurizations for downstream tasks. 
One notable such featurization is the persistent homology transform (PHT) @turner2014persistent, an injective transform which filters a triangulated space $cal(X)$ embedded in $bb(R)^d$ via the sublevel set parameterization $cal(X)(v)_r = {x in cal(X) : x dot v <= r }$, where $v in S^(d-1)$. The intuition here is that by "looking" at the sublevel-set persistence of the shape $cal(X)$ from all possible directions $v in S^1$, one can recover enough information to fully reconstruct $cal(X)$ from the collection of diagrams alone. 


#figure([
	#image("../images/mpeg7_heattrace.png", width: 65%)],
	placement: top, 
  caption: [
		Spectral signatures generated from filtering combinatorial Laplacians of MPEG7 2D shape datasets using the directional transform. By using the relaxation from @eq:heat_sf, these signatures can be interpreted as combinations of heat traces over the interval $[0, 2 pi)$. Observe that highly similar shapes generate curves that maintain a large degree of intra-class similarity under the $ell_2$ norm whereas outliers are detected and retain dissimilarity between classes (e.g. the bent bone in the black class exhibits higher peaks in it signature compared to its class, but is highly class-similar.)
  ]
)<fig:mpeg7_curves>

Aside from sparking an inverse theory for persistence, the injectivity of the PHT also provides the foundation necessary for equipping shape spaces with bona fide distance metrics. For example, one such metric is given by the integrated Wasserstein distance $op("dist")_(cal(X))$ between the sublevel-set persistence diagrams $dgm_p (X, v)$: 

$ op("dist")_(cal(X)) (X, X') = sum_(p=0)^d integral_(S^(d-1)) d_W (dgm_p (X, v), dgm_p (X', v)) d v $

where $X, X' in cal(X)$. In most machine learning applications, distance metrics are often difficult---if not intractable--to analytically determine; instead, most practitioners either craft or learn task-specific pseudometrics that are easy to compute and discriminative enough for the given application. 

We seek to learn shape-sensitive, discriminative pseudometrics from weaker persistence invariants. Towards this end, for some fixed simplicial complex $K$ and pair $(a,b) in Delta_+$, consider the following parameterization over $S^(d-1)$: 

$ beta_p^(a,b)( X, v) =  abs({ (a',b') in dgm_p (K, v) : a' <= a, b < b' }) $

When $d = 2$, this is a piecewise-constant function which is periodic over the interval $[0, 2 pi)$. The collection of all such curves over $Delta_+$ characterizes all the information in PHT, thus it is a natural candidate hypothesis space for learning algorithms. However, some choices of pairs $(a,b) in Delta_+$ may yield functions which have little-to-no information (e.g. they are constant), suggesting that some choices of $(a,b) in Delta_+$ may be more useful than others for the purpose of classification. 

To demonstrate another application of our methodology for shape classification, we apply our spectral relaxation to the MPEG-7 shape matching data set from @bai2009learning. 
To build a classifier, we start by generating curves $cal(B)_(a,b) = { hat(beta)_p^(a,b)(K, v) }_(v in S^1)$
for random choice $(a,b) in Delta_+$. These curves are used to build a hypothesis function $h_ast (x)$, for use in an ensemble model $H$: 

$ H(x) = alpha_1 h_1 (x) + alpha_2 h_2 (x) + dots + alpha_t h_t (x) $

To determine the coefficients $alpha_ast$, we use the classical AdaBoost method, sampling new hypothesis functions $h_ast$ randomly a bounded subset of $Delta_+$ with uniform probability. We choose the spectral relaxation $phi.alt (lambda, tau) = 1 - exp(-lambda slash tau)$ with a smoothing parameter $tau = 1 dot 10^(-1)$. Following @eq:heat_sf, the corresponding curves $cal(B)_(a,b)$ may be interpreted as combinations of _heat kernel traces_, which are known to be useful for characterizing graphs @xiao2009graph. 

In @fig:mpeg7_curves, we plot the curves representing a hypothesis function generated for the ensemble classifier. Observe that highly similar shapes generate curves that maintain a large degree of intra-class similarity under the $ell_2$ norm whereas outliers are detected and retain dissimilarity between classes (e.g. the bent bone in the black class exhibits higher peaks in it signature compared to its class, but is highly class-similar.)

// opt for a simple ensemble model using AdaBoost with the : 

// $ H(K) = alpha_1 dot hat(beta)_p^(a_1,b_2)(K)$

// Conceptually, our approach is to view the PHT as a vineyard, and then reinterpret the problem of constructing it as constructing $mu_p^R(K, f_v)$, for all $v in S^1$ and all $R subset Delta_+$ where . 
// The theory from Chazal states that the rank function contains all the information about the diagram and vice-versa---thus, we may reconstruct 
// However, despite their power, methods such as the PHT can be computationally prohibitive, driving the need for practical approximations that retain essential topological information.



// === Manifold detection from image patches <manifold-detection-from-image-patches>
// A common hypothesis is that high dimensional data tend to lie in the vicinity of an embedded, low dimensional manifold or topological space. An exemplary demonstration of this is given in the analysis by Lee et al. @lee2003nonlinear, who explored the space of high-contrast patches extracted from Hans van Hateren’s still image collection,#footnote[See #link("http://bethgelab.org/datasets/vanhateren/") for details on the image collection.] which consists of $approx 4 upright(",") 000$ monochrome images depicting various areas outside Groningen (Holland). Originally motivated by discerning whether there existed clear qualitative differences in the distributions of patches extracted from images of different modalities, such as optical and range images, Lee et al. @lee2003nonlinear were interested in exploring how high-contrast $3 times 3$ image patches were distributed in pixel-space with respect to predicted spaces and manifolds. Formally, they measured contrast using a discrete version of the scale-invariant Dirichlet semi-norm: $ norm(x)_D = sqrt(sum_(i tilde.op j) (x_i - x_j)^2) = sqrt(x^T D x) $ where $D$ is a fixed matrix whose quadratic form $x^T D x$ applied to an image $x in bb(R)^9$ is proportional to the sum of the differences between each pixels 4 connected neighbors (given above by the relation $i tilde.op j$). By mean-centering, contrast normalizing, and"whitening" the data via the Discrete Cosine Transform (DCT), they show a convenient basis for $D$ may be obtained via an expansion of 8 certain non-constant eigenvectors:

// #image("dct_basis_trimmed.png", width: 80%)

// Since these images are scale-invariant, the expansion of these basis vectors spans the 7-sphere, $S^7 subset bb(R)^8$. Using a Voronoi cell decomposition of the data, their distribution analysis suggested that the majority of data points concentrated in a few high-density regions.

// In follow-up work, Carlsson et al. @carlsson2008local used persistent homology to find the distribution of high-contrast $3 times 3$ patches is actually well-approximated by a Klein bottle $cal(M)$—around 60% of the high-contrast patches from the still image data set lie within a small neighborhood around $cal(M)$ accounting for only 21% of the 7-sphere’s volume. Though a certainly remarkable result, if one was not aware of the analysis done by @lee2003nonlinear @lee2003nonlinear, it would not be immediately clear a priori how to reproduce the discovery in the more general setting; e.g. how does one determine which topological space is a viable model for image patches? Indeed, armed with both efficient persistent homology software and refined topological intuition, Carlsson still needed to perform extensive point-sampling, preprocessing, and model fitting techniques in order to substantiate the Klein bottle was an appropriate space for the distribution of $3 times 3$ patches @carlsson2008local. One of the (many) potential applications of #emph[multi-parameter persistence]—which filters the data along multiple dimensions—is to eliminate the necessity of such extensive preprocessing, thereby dramatically improving the practical ability of performing homological inference on noisy data.

// #image("hilbert_unmarked.png", width: 40%) #image("pers5.png", width: 40%)