import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase27_a_taxonomy_redesign_v2_rules as rules


class Phase27ATaxonomyRulesTest(unittest.TestCase):
    def test_safe_float_uses_default_for_missing_or_bad_values(self):
        self.assertEqual(rules.safe_float(None, 0.25), 0.25)
        self.assertEqual(rules.safe_float("", 0.25), 0.25)
        self.assertEqual(rules.safe_float("not-a-number", 0.25), 0.25)
        self.assertEqual(rules.safe_float("0.5", 0.25), 0.5)

    def test_default_thresholds_are_conservative_contract(self):
        self.assertEqual(
            rules.default_thresholds(),
            {
                "evidence_sufficiency_high": 0.67,
                "evidence_sufficiency_low": 0.33,
                "observability_low": 0.33,
                "conflict_high": 0.67,
                "baseline_difficulty_high": 0.67,
                "stress_sensitivity_high": 0.67,
                "tail_risk_high": 0.67,
                "control_stability_high": 0.67,
                "control_stability_low": 0.33,
            },
        )

    def test_unknown_unvalidated_for_missing_identity_or_required_axes(self):
        base = self._row(canonical_pair_id="")
        self.assertEqual(rules.assign_derived_state(base), "unknown_unvalidated")

        missing_axis = self._row()
        missing_axis.pop("heading_observability_score")
        self.assertEqual(rules.assign_derived_state(missing_axis), "unknown_unvalidated")

    def test_low_observable_and_ambiguous_unreliable_outrank_hard(self):
        low_observable = self._row(
            evidence_sufficiency_score=0.9,
            heading_observability_score=0.2,
            range_observability_score=0.9,
            baseline_error_score=0.9,
            control_stability_score=0.1,
        )
        self.assertEqual(rules.assign_derived_state(low_observable), "low_observable")

        ambiguous = self._row(
            evidence_sufficiency_score=0.2,
            heading_observability_score=0.9,
            range_observability_score=0.9,
            semantic_geometric_conflict_score=0.9,
            baseline_error_score=0.9,
            control_stability_score=0.1,
        )
        self.assertEqual(rules.assign_derived_state(ambiguous), "ambiguous_unreliable")

    def test_stress_sensitive_control_outranks_stable_anchor(self):
        row = self._row(
            evidence_sufficiency_score=0.9,
            match_sufficiency_score=0.9,
            heading_observability_score=0.9,
            range_observability_score=0.9,
            semantic_geometric_conflict_score=0.1,
            baseline_error_score=0.1,
            stress_sensitivity_score=0.9,
            ambiguity_tail_risk_score=0.1,
            control_stability_score=0.9,
        )
        self.assertEqual(rules.assign_derived_state(row), "stress_sensitive_control")

    def test_conflict_candidate_is_not_automatically_hard(self):
        row = self._row(
            evidence_sufficiency_score=0.9,
            match_sufficiency_score=0.9,
            heading_observability_score=0.9,
            range_observability_score=0.9,
            semantic_geometric_conflict_score=0.9,
            baseline_error_score=0.9,
            control_stability_score=0.9,
        )
        self.assertEqual(rules.assign_derived_state(row), "conflict_candidate")
        self.assertEqual(
            rules.assign_training_readiness_verdict("conflict_candidate"),
            "READY_FOR_CORRESPONDENCE_ROUTING_CANDIDATE",
        )

    def test_state_to_training_readiness_verdicts(self):
        expected = {
            "stable_control_anchor": "READY_FOR_ANCHOR_CANDIDATE",
            "evidence_sufficient_hard": "READY_FOR_HARD_TRAINING_CANDIDATE",
            "conflict_candidate": "READY_FOR_CORRESPONDENCE_ROUTING_CANDIDATE",
            "low_observable": "QUARANTINE_OR_WEAK_SUPERVISION_CANDIDATE",
            "ambiguous_unreliable": "QUARANTINE_OR_WEAK_SUPERVISION_CANDIDATE",
            "stress_sensitive_control": "ANALYSIS_ONLY",
            "unknown_unvalidated": "NOT_READY",
        }
        for state, verdict in expected.items():
            with self.subTest(state=state):
                self.assertEqual(rules.assign_training_readiness_verdict(state), verdict)

    def test_assign_rows_supports_lists_and_dataframe_like_records(self):
        rows = [self._row(canonical_pair_id="a"), self._row(canonical_pair_id="b", baseline_error_score=0.9)]
        assigned = rules.assign_rows(rows)
        self.assertEqual(assigned[0]["derived_state"], "stable_control_anchor")
        self.assertEqual(assigned[0]["training_readiness_verdict"], "READY_FOR_ANCHOR_CANDIDATE")
        self.assertEqual(assigned[1]["derived_state"], "evidence_sufficient_hard")
        self.assertNotIn("derived_state", rows[0])

        frame_like = RecordsFrame(rows)
        assigned_from_frame = rules.assign_rows(frame_like)
        self.assertEqual(len(assigned_from_frame), 2)
        self.assertEqual(assigned_from_frame[1]["training_readiness_verdict"], "READY_FOR_HARD_TRAINING_CANDIDATE")

    def _row(self, **overrides):
        row = {
            "canonical_pair_id": "pair-1",
            "evidence_sufficiency_score": 0.8,
            "heading_observability_score": 0.8,
            "range_observability_score": 0.8,
            "semantic_geometric_conflict_score": 0.1,
            "match_sufficiency_score": 0.8,
            "layout_scale_risk_score": 0.1,
            "augmentation_consistency_score": 0.8,
            "baseline_error_score": 0.1,
            "stress_sensitivity_score": 0.1,
            "ambiguity_tail_risk_score": 0.1,
            "control_stability_score": 0.8,
        }
        row.update(overrides)
        return row


class RecordsFrame:
    def __init__(self, records):
        self._records = records

    def to_dict(self, orient):
        if orient != "records":
            raise AssertionError("assign_rows should request records orientation")
        return [dict(row) for row in self._records]


if __name__ == "__main__":
    unittest.main()
