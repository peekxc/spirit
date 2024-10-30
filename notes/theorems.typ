
#import "@preview/ctheorems:1.1.3": *
#show: thmrules.with(qed-symbol: $square$)

#let thmplain = thmbox.with(
	padding: (top: 0em, bottom: 0em),
	breakable: true,
	inset: (top: 0em, left: 0.0em, right: 0.0em),
	namefmt: name => emph([(#name)]),
	titlefmt: emph,
)

#let thm_envs = ("Lemma", "Proposition", "Corollary", "Theorem")
#let (lemma, proposition, corollary, theorem) = thm_envs.map((thm_id) =>
	thmenv(
		thm_id,   // identifier
		none,      // base - do not attach, count globally
		none,      // base_level - use the base as-is
		(name, number, body, color: black) => {
			set align(left)
			set par(justify: true)
			let prefix = [*#thm_id #number*]
			if type(name) == str {
				prefix += if name.len() > 0 [* (#name)*] else []
			} else if type(name) == content {
				prefix += if name.fields().len() > 0 [* (#name)*] else []	
			}
			prefix + text(": ", weight: "bold") + body
			// v(-0.2em)
		}
	).with(numbering: "1"))

#let definition = thmplain("definition", "Definition").with(numbering: none)
#let example = thmplain("example", "Example").with(numbering: none)
#let remark = thmplain("remark", "Remark").with(numbering: none)
#let proof = thmproof("proof", "Proof")