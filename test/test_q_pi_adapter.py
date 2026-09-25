"""Focused tests for the isolated native-q_pi adapter experiment."""

from dataclasses import replace
import math

import pytest

from examples.q_pi_adapter import (
    InvalidQPiEvidence,
    QPiSnapshot,
    capture_native_q_pi,
    policy_id,
    policy_table_id,
    validate_and_derive,
)


def test_native_capture_has_valid_ordered_telemetry():
    evidence = capture_native_q_pi()
    assert evidence.policy_table == (
        ((0,), (0,)),
        ((0,), (1,)),
        ((1,), (0,)),
        ((1,), (1,)),
    )
    assert evidence.policy_ids == tuple(policy_id(p) for p in evidence.policy_table)
    assert evidence.policy_table_id == policy_table_id(evidence.policy_table)
    result = validate_and_derive(evidence)
    assert [item.sequence for item in result] == [0, 1]
    assert all(item.active_path_count == 4 for item in result)
    assert all(0.0 <= item.normalized_entropy <= 1.0 for item in result)


def test_tied_posterior_reports_all_tied_policy_ids_in_order():
    evidence = capture_native_q_pi()
    tied = replace(
        evidence,
        snapshots=(QPiSnapshot(0, evidence.policy_table_id, (0.5, 0.5, 0.0, 0.0)),),
    )
    item = validate_and_derive(tied)[0]
    assert item.is_tie
    assert item.tied_policy_ids == evidence.policy_ids[:2]
    assert item.argmax_policy_id == evidence.policy_ids[0]
    assert item.active_path_count == 2
    assert math.isclose(item.normalized_entropy, 1.0)


@pytest.mark.parametrize(
    "q_pi",
    [None, (1.0,), (0.5, -0.1, 0.3, 0.3), (math.nan, 0.0, 0.0, 1.0), (0.1,) * 4],
)
def test_missing_or_malformed_q_pi_is_rejected(q_pi):
    evidence = capture_native_q_pi()
    malformed = replace(
        evidence, snapshots=(QPiSnapshot(0, evidence.policy_table_id, q_pi),)
    )
    with pytest.raises(InvalidQPiEvidence):
        validate_and_derive(malformed)


def test_snapshot_ordering_error_is_rejected():
    evidence = capture_native_q_pi()
    with pytest.raises(InvalidQPiEvidence, match="contiguous"):
        validate_and_derive(replace(evidence, snapshots=evidence.snapshots[::-1]))


def test_policy_ordering_error_is_rejected():
    evidence = capture_native_q_pi()
    reordered = replace(evidence, policy_table=evidence.policy_table[::-1])
    with pytest.raises(InvalidQPiEvidence, match="policy IDs"):
        validate_and_derive(reordered)


def test_policy_population_change_is_rejected_without_repair():
    evidence = capture_native_q_pi()
    shortened = evidence.policy_table[:-1]
    changed = replace(
        evidence,
        policy_table=shortened,
        policy_ids=tuple(policy_id(p) for p in shortened),
        policy_table_id=policy_table_id(shortened),
    )
    with pytest.raises(InvalidQPiEvidence, match="changed"):
        validate_and_derive(changed)
