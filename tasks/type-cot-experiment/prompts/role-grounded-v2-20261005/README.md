# Role-grounded Type-CoT prompt draft

This independently saved prompt rewrites the task and all ten event-type
descriptions. It supplies the complete existing role lists and explicitly
requests concise explanations grounded in argument text and semantic roles.
It is a reviewable draft, not a launched experiment or an updated dataset.

Task scope remains target-sentence triggers with full-abstract context and
cross-sentence arguments. The final output remains a type multiset inventory;
reasoning headings match the existing CoT corpus. Gold events/arguments/types
must never be inserted into the model's input. Existing gold-guided CoT answers
are training targets, not inference inputs.

Role names and ordering were checked against
`tasks/stage2-event-tree/curriculum/prompt_template.txt`.
Definitions are a clearer interpretation of the existing schema, not a new
annotation policy. Sparse annotated roles must remain sparse: a schema slot
is not a requirement to add an unannotated argument to an existing answer.

Before building a new dataset, inspect existing CoT completions for agreement
with these instructions, especially counting separate relations in one clause
and the distinctions WKS/MDS/PRP and FIN/CMP/FAC. Rewriting the input does not
repair an inaccurate explanation in a training target. Preserve old files and
record prompt/data hashes for any new run. No accuracy improvement is assumed.
