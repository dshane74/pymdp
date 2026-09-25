# Native `q_pi` adapter experiment

`examples/q_pi_adapter.py` is an isolated, deterministic experiment around the
production `Agent.infer_states` and `Agent.infer_policies` APIs. It uses one
two-state factor, one two-observation modality, two controls, and a two-step
planning horizon.

The raw record retains the exact ordered policy table and ordered `q_pi`
snapshots. SHA-256 IDs cover each complete nested policy and the complete
ordered table. Validation is fail-closed: missing, malformed, reordered, or
population-mismatched evidence raises `InvalidQPiEvidence`; values are never
filled, clipped, renormalized, or reordered.

Only after validation does the adapter calculate active path count, Shannon
path-weight entropy, entropy normalized by the active population, maximum model
weight, the first argmax policy ID, and explicit tie status/IDs. Raw evidence
and derived records are separate immutable dataclasses.

Run the demonstration with `python -m examples.q_pi_adapter`, or run its focused
tests with `pytest -q test/test_q_pi_adapter.py`.

This experiment does not produce or imply prediction/falsification statistics,
reinforcement, drift, DMA, detector verdicts, or safety conclusions.
