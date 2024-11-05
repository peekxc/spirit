#import "@preview/arkheion:0.1.0": arkheion, arkheion-appendices
#import "@preview/lovelace:0.3.0": *
#import "math_ops.typ": *
#import "theorems.typ": *
#show: thmrules 

// arxiv prefers us-letter 
#set page("us-letter", margin: (y: 1in, x:0.84in))

// Prefer (1) for referencing equations
#set math.equation(numbering: "(1)")
#show ref: it => {
	let eq = math.equation
	let el = it.element
	if el != none and el.func() == eq {
		// Override equation references.
		numbering(el.numbering, ..counter(eq).at(el.location()) )
	} else {
		it // Other references as usual.
	}
}

// font size 
#set text(size: 10pt)
#show figure.caption: it => [
	#set text(size: 9.0pt)
	#set align(left)
	#it 
]

#show: arkheion.with(
  title: "Spectral relaxation of the persistence rank invariant",
  authors: (
    (name: "Matt Piekenbrock", email: "piekenbrock.m@northeastern.edu", affiliation: "Northeastern University"), // orcid: "0000-0000-0000-0000"
    (name: "Jose Perea", email: "j.pereabenitez@northeastern.edu", affiliation: "Northeastern University"),
  ),
  abstract: align(left, [
		Using a duality result between persistence diagrams and persistence measures, we introduce a framework for constructing families of continuous relaxations of the persistent rank invariant for parametrized families of persistence vector spaces indexed over the real line. Like the rank invariant, these families obey inclusion-exclusion, derive from simplicial boundary operators, and encode all the information needed to construct a persistence diagram. 
		Unlike the rank invariant, these spectrally-derived families enjoy a number of stability and continuity properties, such as smoothness and differentiability over the positive semi-definite cone. 
		Surprisingly, by leveraging a connection between stochastic Lanczos quadrature and implicit trace estimation, our proposed relaxation enables the matrix-free, iterative approximation scheme for all of the persistence invariants---Betti numbers, persistent pairs, and cycle representatives---in linear space and cubic time in the size of the filtration. 
	]),
  keywords: ("Topological Data Analysis", "Persistent Homology", "Matrix functions"),
  // date: "May 16, 2023",
)

// #set par(spacing: 1.0em)
#show math.equation: set block(above: 1.20em, below: 1.20em)

// Section 1
#include "includes/introduction.typ"

// Section 2
#include "includes/notation.typ"
#include "includes/background.typ"

// Section 3
#include "includes/spectral.typ"

// Section 4
#include "includes/computation.typ"

// Section 5
#include "includes/applications.typ"

// Section 6
#include "includes/conclusion.typ"

#pagebreak(weak: false)

// References
#bibliography(
	"references.bib", 
	title: "References"
)
#pagebreak(weak: false)

// Appendix
// #show: arkheion-appendices
#include "includes/appendix.typ"