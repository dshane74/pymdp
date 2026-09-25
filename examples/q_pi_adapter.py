"""Isolated, fail-closed telemetry adapter for pymdp's native ``q_pi``.

This module deliberately does not change the agent or interpret the posterior as
anything other than weights over the agent's ordered policy table.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Optional

import jax.numpy as jnp

from pymdp.agent import Agent


Policy = tuple[tuple[int, ...], ...]
PolicyTable = tuple[Policy, ...]


class InvalidQPiEvidence(ValueError):
    """Raised when raw evidence is absent, inconsistent, or malformed."""


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_id(policy: Policy) -> str:
    """Return an ID covering every time step and control factor in a policy."""
    return _stable_hash(policy)


def policy_table_id(policy_table: PolicyTable) -> str:
    """Return an ID covering the complete ordered policy population."""
    return _stable_hash(policy_table)


@dataclass(frozen=True)
class QPiSnapshot:
    sequence: int
    policy_table_id: str
    q_pi: Optional[tuple[float, ...]]


@dataclass(frozen=True)
class QPiEvidence:
    policy_table: PolicyTable
    policy_ids: tuple[str, ...]
    policy_table_id: str
    snapshots: tuple[QPiSnapshot, ...]


@dataclass(frozen=True)
class QPiDerived:
    sequence: int
    active_path_count: int
    path_weight_entropy: float
    normalized_entropy: float
    max_model_weight: float
    argmax_policy_id: str
    is_tie: bool
    tied_policy_ids: tuple[str, ...]


def validate_and_derive(evidence: QPiEvidence) -> tuple[QPiDerived, ...]:
    """Validate raw evidence without repairing it, then derive a small metric set."""
    count = len(evidence.policy_table)
    expected_ids = tuple(policy_id(policy) for policy in evidence.policy_table)
    expected_table_id = policy_table_id(evidence.policy_table)
    if count == 0:
        raise InvalidQPiEvidence("policy table is empty")
    if not evidence.snapshots:
        raise InvalidQPiEvidence("q_pi snapshots are missing")
    if evidence.policy_ids != expected_ids or evidence.policy_table_id != expected_table_id:
        raise InvalidQPiEvidence("policy IDs do not match the ordered policy table")

    derived = []
    for expected_sequence, snapshot in enumerate(evidence.snapshots):
        if snapshot.sequence != expected_sequence:
            raise InvalidQPiEvidence("snapshot sequence numbers are not contiguous from zero")
        if snapshot.policy_table_id != expected_table_id:
            raise InvalidQPiEvidence("policy table changed during capture")
        if snapshot.q_pi is None:
            raise InvalidQPiEvidence("q_pi is missing")
        if len(snapshot.q_pi) != count:
            raise InvalidQPiEvidence("q_pi length does not match the policy count")
        if any(
            not isinstance(weight, Real)
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0.0
            for weight in snapshot.q_pi
        ):
            raise InvalidQPiEvidence("q_pi contains a non-finite or negative value")
        if not math.isclose(math.fsum(snapshot.q_pi), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise InvalidQPiEvidence("q_pi does not sum to one")

        active = tuple(weight for weight in snapshot.q_pi if weight > 0.0)
        entropy = -math.fsum(weight * math.log(weight) for weight in active)
        normalized = entropy / math.log(len(active)) if len(active) > 1 else 0.0
        maximum = max(snapshot.q_pi)
        tied_indices = tuple(
            index for index, weight in enumerate(snapshot.q_pi) if weight == maximum
        )
        derived.append(
            QPiDerived(
                sequence=snapshot.sequence,
                active_path_count=len(active),
                path_weight_entropy=entropy,
                normalized_entropy=normalized,
                max_model_weight=maximum,
                argmax_policy_id=evidence.policy_ids[tied_indices[0]],
                is_tie=len(tied_indices) > 1,
                tied_policy_ids=tuple(evidence.policy_ids[index] for index in tied_indices),
            )
        )
    return tuple(derived)


def capture_native_q_pi() -> QPiEvidence:
    """Run the deterministic 2-state/2-observation/2-control experiment."""
    identity = jnp.eye(2)
    switch = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    agent = Agent(
        A=[identity],
        B=[jnp.stack((identity, switch), axis=-1)],
        C=[jnp.array([0.0, 2.0])],
        num_controls=[2],
        policy_len=2,
    )
    table: PolicyTable = tuple(
        tuple(tuple(int(control) for control in step) for step in policy)
        for policy in agent.policies.policy_arr.tolist()
    )
    table_id = policy_table_id(table)
    snapshots = []
    for sequence, observation in enumerate((0, 1)):
        beliefs = agent.infer_states(
            [jnp.array([observation])], empirical_prior=agent.D
        )
        q_pi, _ = agent.infer_policies(beliefs)
        snapshots.append(
            QPiSnapshot(sequence, table_id, tuple(float(value) for value in q_pi[0]))
        )
    return QPiEvidence(
        policy_table=table,
        policy_ids=tuple(policy_id(policy) for policy in table),
        policy_table_id=table_id,
        snapshots=tuple(snapshots),
    )


if __name__ == "__main__":
    raw = capture_native_q_pi()
    print(
        json.dumps(
            {"raw": asdict(raw), "derived": [asdict(x) for x in validate_and_derive(raw)]},
            indent=2,
        )
    )
