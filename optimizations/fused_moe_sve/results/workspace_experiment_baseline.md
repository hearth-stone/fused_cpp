# Current offline experiment baseline

For new Arm NUMA3 planner/cost-model hardware experiments, explicitly pass
`--experiment-baseline optimizations/fused_moe_sve/profiles/workspace_numa3_80c.json`
to `bench_bounded_order_extension.py measure`. The profile selects a fixed,
preallocated/pretouched2048-token FP32 route workspace and freezes extension,
calibration and workspace-helper identities. Actual packed backend Ntile16 is
checked and recorded. Production defaults are unchanged.

Protocol: CPUs240–319/membind3,E256/H4096/F512,TopK6,2048tokens,4 packed copies,
216MiB scrub,5 warmups+31 paired effective rounds,no phase trace,NaN-poisoned
output correctness before timing. Workspace initialization is separate. Growth
is low priority and unsupported; capacity mismatch fails rather than reallocating.

Each new session records `output_lifecycle`, `experiment_baseline_sha256`,
`workspace_max_tokens`, helper identity and actual `backend_n_tile`. Analysis
rejects mismatched session identities and mismatches against a profile-bound
frontier. Prior explicit workspace sessions without this new profile field retain
their original metadata; they may be cited as historical evidence, not silently
relabeled or mixed into a new confirmatory session pair. Allocation-per-call data
remain a separate regime. Historical records are never overwritten to fix labels.

Frozen v8 may be used for point predictions and diagnostic ranking. Its existing
pairwise residual report does NOT authorize automatic acceptance/pruning under
this lifecycle. Until an independently validated workspace-specific report exists,
keep partial-order outputs diagnostic and retain diverse candidates/hardware rerank.
Never fit old total residuals as bandwidth penalties or fit these evaluation
frontiers to make a new report pass.

LNS remains a candidate generator, and VND remains useful for weaker starts.
Anchor/elite confirmation must compare against the retained strong baseline in
the same sessions. A positive move relative to a weaker parent is not an anchor
promotion. Accept only both-session >2% paired median gain with positive P10;
among near-equal candidates retain elites without claiming an exact optimum.

Implementation files are Lab-only and explicit opt-in. Changes to the profile or
helper require a distinct recorded identity and validation; do not edit historical
session artifacts to match them. No production promotion is implied by this file.
