import unittest

from scripts import phase27_a_taxonomy_redesign_v3_candidates as candidates


class CandidateRulesTest(unittest.TestCase):
    def test_high_evidence_creates_evidence_sufficient(self):
        row = {
            "evidence_sufficiency_score": "0.8",
            "heading_observability_score": "0.6",
            "range_observability_score": "0.4",
            "semantic_geometric_conflict_score": "0.1",
            "match_sufficiency_score": "0.2",
            "ambiguity_tail_risk_score": "0.1",
            "control_stability_score": "0.8",
            "layout_scale_risk_score": "0.1",
        }
        out = candidates.derive_layer1_candidates(row)
        self.assertTrue(out["evidence_sufficient_candidate"])
        self.assertTrue(out["heading_observable_candidate"])
        self.assertFalse(out["range_observable_candidate"])

    def test_low_observable_vetoes_control_but_not_other_flags(self):
        row = {
            "evidence_sufficiency_score": "0.1",
            "heading_observability_score": "0.6",
            "range_observability_score": "0.6",
            "semantic_geometric_conflict_score": "0.6",
            "match_sufficiency_score": "0.7",
            "ambiguity_tail_risk_score": "0.2",
            "control_stability_score": "0.9",
            "layout_scale_risk_score": "0.1",
            "low_observable_flag": "1",
        }
        out = candidates.derive_layer1_candidates(row)
        self.assertTrue(out["low_observable_candidate"])
        self.assertTrue(out["semantic_geometric_conflict_candidate"])
        self.assertFalse(out["control_candidate"])

    def test_conflict_coexists_with_evidence(self):
        row = {
            "evidence_sufficiency_score": "0.7",
            "heading_observability_score": "0.6",
            "range_observability_score": "0.6",
            "semantic_geometric_conflict_score": "0.8",
            "match_sufficiency_score": "0.7",
            "ambiguity_tail_risk_score": "0.8",
            "control_stability_score": "0.2",
            "layout_scale_risk_score": "0.1",
        }
        out = candidates.derive_layer1_candidates(row)
        self.assertTrue(out["evidence_sufficient_candidate"])
        self.assertTrue(out["semantic_geometric_conflict_candidate"])
        self.assertTrue(out["local_alignment_needed_candidate"])
        self.assertTrue(out["multi_modal_ambiguous"])


if __name__ == "__main__":
    unittest.main()
