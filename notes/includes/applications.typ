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

In many time series applications, detecting periodic behavior can prove a difficult yet useful thing to estimate. For example, in medical images contexts, it is necessary to preprocess the data to remove noise and outliers that otherwise obscure the presence of recurrent behavior. 

#figure([#image("../images/sw1pers_mu.png", width: 100%)],
  caption: [
    SW1Pers
  ]
)<fig:sw1pers_spirit>

// We now formalize this process. Assume the set of topological constraints are given as input via a set of pairs $brace.l thin (R_1 , c_1)) , lr((R_2 , c_2) , dots.h , (R_h , c_h) thin brace.r$, where each $R_i subset Delta_+$ prescribing rectangular areas of wherein a multiplicity constraint on the persistence is imposed. We seek to find the minimal size mesh $K$

// $ min_(alpha in bb(R)) quad & hash thin a r d) lr((K_alpha^p))\
// upright("s.t.") quad & mu_p^(R_i) (K_alpha , f_alpha) = c_i , quad forall #h(0em) R_i in cal(R) $ In other words, the set
// 
// 
// 
// 
// 6

// === Manifold detection from image patches <manifold-detection-from-image-patches>
// A common hypothesis is that high dimensional data tend to lie in the vicinity of an embedded, low dimensional manifold or topological space. An exemplary demonstration of this is given in the analysis by Lee et al. @lee2003nonlinear, who explored the space of high-contrast patches extracted from Hans van Hateren’s still image collection,#footnote[See #link("http://bethgelab.org/datasets/vanhateren/") for details on the image collection.] which consists of $approx 4 upright(",") 000$ monochrome images depicting various areas outside Groningen (Holland). Originally motivated by discerning whether there existed clear qualitative differences in the distributions of patches extracted from images of different modalities, such as optical and range images, Lee et al. @lee2003nonlinear were interested in exploring how high-contrast $3 times 3$ image patches were distributed in pixel-space with respect to predicted spaces and manifolds. Formally, they measured contrast using a discrete version of the scale-invariant Dirichlet semi-norm: $ norm(x)_D = sqrt(sum_(i tilde.op j) (x_i - x_j)^2) = sqrt(x^T D x) $ where $D$ is a fixed matrix whose quadratic form $x^T D x$ applied to an image $x in bb(R)^9$ is proportional to the sum of the differences between each pixels 4 connected neighbors (given above by the relation $i tilde.op j$). By mean-centering, contrast normalizing, and"whitening" the data via the Discrete Cosine Transform (DCT), they show a convenient basis for $D$ may be obtained via an expansion of 8 certain non-constant eigenvectors:

// #image("dct_basis_trimmed.png", width: 80%)

// Since these images are scale-invariant, the expansion of these basis vectors spans the 7-sphere, $S^7 subset bb(R)^8$. Using a Voronoi cell decomposition of the data, their distribution analysis suggested that the majority of data points concentrated in a few high-density regions.

// In follow-up work, Carlsson et al. @carlsson2008local used persistent homology to find the distribution of high-contrast $3 times 3$ patches is actually well-approximated by a Klein bottle $cal(M)$—around 60% of the high-contrast patches from the still image data set lie within a small neighborhood around $cal(M)$ accounting for only 21% of the 7-sphere’s volume. Though a certainly remarkable result, if one was not aware of the analysis done by @lee2003nonlinear @lee2003nonlinear, it would not be immediately clear a priori how to reproduce the discovery in the more general setting; e.g. how does one determine which topological space is a viable model for image patches? Indeed, armed with both efficient persistent homology software and refined topological intuition, Carlsson still needed to perform extensive point-sampling, preprocessing, and model fitting techniques in order to substantiate the Klein bottle was an appropriate space for the distribution of $3 times 3$ patches @carlsson2008local. One of the (many) potential applications of #emph[multi-parameter persistence]—which filters the data along multiple dimensions—is to eliminate the necessity of such extensive preprocessing, thereby dramatically improving the practical ability of performing homological inference on noisy data.

// #image("hilbert_unmarked.png", width: 40%) #image("pers5.png", width: 40%)