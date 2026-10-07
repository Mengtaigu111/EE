# Validation gates

Approved recipe: plan.md; frozen inputs: manifest.json.

CPU final checks: cpu-online-tests-final.log has 56 passing tests and 21 passing
subtests; cpu-type-tests.log has 5 passing tests. This includes actual small
PEFT Trainer checkpoint/early-stop integration, handoff state capture, explicit
cadence, independent feedback replay, and test/LR verification regression tests.
Warnings are from the small GPT2/PEFT test fixtures, not GPU training failures.

The initial combined pytest collection imported same-name legacy `build_dataset`
and `evaluate` modules from the tree suite into the CoT suite and caused two
ImportErrors (cpu-tests.log). The suites were rerun in separate Python processes,
matching the actual two-stage subprocess isolation. Both separate suites pass;
no historical modules were modified to accommodate test collection.

Independent specification/data review: review.md. Independent quality review:
quality-review.md. Quality review found missing final test raw→merged consistency,
completion/hash and dev-best binding checks plus incomplete LR group validation.
Those were fixed with regression tests; reviewer confirmed findings closed.

First GPU smoke completed with exit0 and full two-stage verification, including
14 exact independent-reload outputs. Its source snapshot is retained. Verifier
hardening was applied only after that smoke exited, and a fresh final-source
smoke is required before formal launch. See smoke-launch-first.json and
smoke-launch.json for immutable run paths and current launch metadata.

No test scores are used for training or checkpoint selection. CoT dev diagnostics
do not change the fixed one-epoch CoT budget or final checkpoint handoff.

Final-source smoke completed exit0 and passed a fresh independent verify.py
replay; checkpoint4 reload matches all14 raw outputs. Evidence:
final-smoke-verification.json. Formal training started in the new tmux recorded
in main-launch.json; actual optimizer updates, finite losses, unchanged hashes,
correct shuffled prefix and a live GPU process were verified. See
main-start-verification.json and launch-report.md. Training is ongoing.
