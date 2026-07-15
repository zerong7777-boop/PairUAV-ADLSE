import unittest

from scripts.phase27_a_evidence_state_v3_calibration_v2 import (
    ADEQUACY_FIELDS,
    AXIS_NAMES,
    BASE_REGIMES,
    CALIBRATION_V2_VERSION,
    FORBIDDEN_CONSTRUCTION_PATTERNS,
    audit_calibration_v2_columns,
    assign_evidence_state_v2,
    canonical_pair_id,
    canonicalize_image_token,
    compute_adequacy,
    compute_calibrated_axes_v2,
    compute_control_centrality,
    compute_layout_risk_axis,
    compute_observability_axis,
    compute_pair_similarity_axis,
    compute_scale_risk_axis,
    fit_calibration_v2,
)


BASE_ROW = {
    "source_split": "train",
    "group_id": "0881",
    "image_a_name": "image-33.jpeg",
    "image_b_name": "image-02.jpeg",
    "image_a_exists": 1,
    "image_b_exists": 1,
    "has_cheap_image_features": 1,
    "brightness_gap_abs": 8,
    "contrast_gap_abs": 4,
    "grayscale_hist_similarity": 0.72,
    "aspect_ratio_gap_abs": 0.06,
    "image_index_gap_abs": 3,
    "image_a_width": 640,
    "image_a_height": 480,
    "image_b_width": 650,
    "image_b_height": 488,
    "cached_scale_balance": 0.95,
    "cached_spatial_entropy": 0.88,
    "cached_anchor_spread": 0.9,
}


def row(**overrides):
    out = dict(BASE_ROW)
    out.update(overrides)
    return out


TRAIN_ROWS = [
    row(pair_id="0881/33_02", brightness_gap_abs=42, contrast_gap_abs=24, grayscale_hist_similarity=0.36, aspect_ratio_gap_abs=0.36, image_index_gap_abs=15, cached_scale_balance=0.55, cached_spatial_entropy=0.45, cached_anchor_spread=0.5),
    row(pair_id="0881/34_02", brightness_gap_abs=12, contrast_gap_abs=7, grayscale_hist_similarity=0.62, aspect_ratio_gap_abs=0.10, image_index_gap_abs=5, image_a_name="image-34.jpeg"),
    row(pair_id="0881/35_02", brightness_gap_abs=8, contrast_gap_abs=4, grayscale_hist_similarity=0.72, aspect_ratio_gap_abs=0.06, image_index_gap_abs=3, image_a_name="image-35.jpeg"),
    row(pair_id="0881/36_02", brightness_gap_abs=1, contrast_gap_abs=1, grayscale_hist_similarity=0.96, aspect_ratio_gap_abs=0.01, image_index_gap_abs=1, image_a_name="image-36.jpeg", cached_scale_balance=0.99, cached_spatial_entropy=0.95, cached_anchor_spread=0.95),
]


class CalibrationV2CoreTests(unittest.TestCase):
    def test_canonical_image_tokens_and_pair_ids(self):
        self.assertEqual("02", canonicalize_image_token("image-02.jpeg"))
        self.assertEqual("02", canonicalize_image_token("image-2"))
        self.assertEqual("0881/33_02", canonical_pair_id({"group_id": "0881", "pair_id": "0881/33_02"}))
        self.assertEqual("0881/33_02", canonical_pair_id({"group_id": "0881", "pair_id": "33_02"}))
        self.assertEqual("0881/33_02", canonical_pair_id({"group_id": "0881", "json_id": "33_02"}))
        self.assertEqual(
            "0881/33_02",
            canonical_pair_id(
                {
                    "group_id": "0881",
                    "image_a_name": "image-33.jpeg",
                    "image_b_name": "image-02.jpeg",
                }
            ),
        )

    def test_audit_rejects_forbidden_construction_columns(self):
        for pattern in FORBIDDEN_CONSTRUCTION_PATTERNS:
            audit = audit_calibration_v2_columns(["pair_id", f"prefix_{pattern}_suffix"])
            self.assertFalse(audit["passed"], pattern)
            self.assertEqual([f"prefix_{pattern}_suffix"], audit["forbidden_columns"])

        with self.assertRaises(ValueError):
            fit_calibration_v2([row(final_score=0.99)])

    def test_constants_are_exact_task_contract(self):
        self.assertEqual("phase27_a_feature_calibration_v2", CALIBRATION_V2_VERSION)
        self.assertEqual(
            [
                "ordinary_control_anchor",
                "high_evidence_anchor",
                "hard_trainable",
                "low_observable",
                "ambiguous_unreliable",
                "unknown_insufficient_features",
            ],
            BASE_REGIMES,
        )
        self.assertEqual(
            [
                "observability_axis",
                "pair_similarity_axis",
                "scale_risk_axis",
                "layout_risk_axis",
                "conflict_risk_axis",
                "control_centrality_score",
            ],
            AXIS_NAMES,
        )
        self.assertEqual(
            [
                "feature_complete",
                "observable_adequate",
                "image_quality_adequate",
                "pair_identity_valid",
                "adequacy_passed",
                "low_observable_reason",
            ],
            ADEQUACY_FIELDS,
        )

    def test_complete_bottom_quantile_row_is_not_low_observable_without_absolute_failure(self):
        calibration = fit_calibration_v2(TRAIN_ROWS)
        bottom_but_adequate = row(
            pair_id="0881/40_02",
            image_a_name="image-40.jpeg",
            brightness_gap_abs=28,
            contrast_gap_abs=18,
            grayscale_hist_similarity=0.5,
            aspect_ratio_gap_abs=0.22,
            image_index_gap_abs=10,
        )
        axes = compute_calibrated_axes_v2(bottom_but_adequate, calibration)
        adequacy = compute_adequacy(bottom_but_adequate, axes)
        assignment = assign_evidence_state_v2(bottom_but_adequate, axes, adequacy, calibration)

        self.assertEqual(1, adequacy["feature_complete"])
        self.assertEqual(1, adequacy["adequacy_passed"])
        self.assertNotEqual("low_observable", assignment["base_regime"])

    def test_risk_axes_are_named_and_higher_means_higher_risk(self):
        low_risk = row()
        high_risk = row(
            grayscale_hist_similarity=0.95,
            aspect_ratio_gap_abs=0.55,
            image_index_gap_abs=18,
            cached_scale_balance=0.2,
            cached_spatial_entropy=0.2,
            cached_anchor_spread=0.2,
        )
        low_axes = {
            "scale_risk_axis": compute_scale_risk_axis(low_risk),
            "layout_risk_axis": compute_layout_risk_axis(low_risk),
            "pair_similarity_axis": compute_pair_similarity_axis(low_risk),
        }
        high_axes = {
            "scale_risk_axis": compute_scale_risk_axis(high_risk),
            "layout_risk_axis": compute_layout_risk_axis(high_risk),
            "pair_similarity_axis": compute_pair_similarity_axis(high_risk),
        }

        self.assertGreater(high_axes["scale_risk_axis"], low_axes["scale_risk_axis"])
        self.assertGreater(high_axes["layout_risk_axis"], low_axes["layout_risk_axis"])
        self.assertGreater(
            compute_calibrated_axes_v2(high_risk, fit_calibration_v2(TRAIN_ROWS))["conflict_risk_axis"],
            compute_calibrated_axes_v2(low_risk, fit_calibration_v2(TRAIN_ROWS))["conflict_risk_axis"],
        )

    def test_control_centrality_and_single_base_regime_output_contract(self):
        calibration = fit_calibration_v2(TRAIN_ROWS)
        ordinary = row(pair_id="0881/34_02", image_a_name="image-34.jpeg", brightness_gap_abs=12, contrast_gap_abs=7, grayscale_hist_similarity=0.62, aspect_ratio_gap_abs=0.10, image_index_gap_abs=5)
        axes = compute_calibrated_axes_v2(ordinary, calibration)
        adequacy = compute_adequacy(ordinary, axes)
        centrality = compute_control_centrality(ordinary, axes, adequacy, calibration)
        assignment = assign_evidence_state_v2(ordinary, axes, adequacy, calibration)

        self.assertIsInstance(centrality, float)
        self.assertEqual("ordinary_control_anchor", assignment["base_regime"])
        self.assertEqual(1, sum(assignment[f"base_{regime}"] for regime in BASE_REGIMES))
        for field in ["canonical_pair_id", "risk_tags", "calibration_version", "calibration_source"]:
            self.assertIn(field, assignment)
        for field in ADEQUACY_FIELDS + AXIS_NAMES:
            self.assertIn(field, assignment)
        self.assertEqual(CALIBRATION_V2_VERSION, assignment["calibration_version"])

    def test_ordinary_anchor_requires_adequacy_low_risk_no_conflict_and_centrality(self):
        calibration = fit_calibration_v2(TRAIN_ROWS)

        inadequate = row(pair_id="0881/41_02", image_a_name="image-41.jpeg", image_a_exists=0)
        axes = compute_calibrated_axes_v2(inadequate, calibration)
        adequacy = compute_adequacy(inadequate, axes)
        self.assertEqual("low_observable", assign_evidence_state_v2(inadequate, axes, adequacy, calibration)["base_regime"])
        self.assertIn("image_missing", adequacy["low_observable_reason"])

        missing_image_a_exists = row(pair_id="0881/45_02", image_a_name="image-45.jpeg")
        missing_image_a_exists.pop("image_a_exists")
        axes = compute_calibrated_axes_v2(missing_image_a_exists, calibration)
        adequacy = compute_adequacy(missing_image_a_exists, axes)
        self.assertEqual(
            "unknown_insufficient_features",
            assign_evidence_state_v2(missing_image_a_exists, axes, adequacy, calibration)["base_regime"],
        )

        missing_image_b_exists = row(pair_id="0881/46_02", image_a_name="image-46.jpeg")
        missing_image_b_exists.pop("image_b_exists")
        axes = compute_calibrated_axes_v2(missing_image_b_exists, calibration)
        adequacy = compute_adequacy(missing_image_b_exists, axes)
        self.assertEqual(
            "unknown_insufficient_features",
            assign_evidence_state_v2(missing_image_b_exists, axes, adequacy, calibration)["base_regime"],
        )

        risky = row(pair_id="0881/42_02", image_a_name="image-42.jpeg", aspect_ratio_gap_abs=0.58, cached_scale_balance=0.25)
        axes = compute_calibrated_axes_v2(risky, calibration)
        adequacy = compute_adequacy(risky, axes)
        self.assertNotEqual("ordinary_control_anchor", assign_evidence_state_v2(risky, axes, adequacy, calibration)["base_regime"])

        conflict = row(pair_id="0881/43_02", image_a_name="image-43.jpeg", grayscale_hist_similarity=0.98, aspect_ratio_gap_abs=0.56, cached_scale_balance=0.2)
        axes = compute_calibrated_axes_v2(conflict, calibration)
        adequacy = compute_adequacy(conflict, axes)
        self.assertEqual("ambiguous_unreliable", assign_evidence_state_v2(conflict, axes, adequacy, calibration)["base_regime"])

        off_center = row(pair_id="0881/44_02", image_a_name="image-44.jpeg", brightness_gap_abs=30, contrast_gap_abs=20, grayscale_hist_similarity=0.48, image_index_gap_abs=12)
        axes = compute_calibrated_axes_v2(off_center, calibration)
        adequacy = compute_adequacy(off_center, axes)
        self.assertNotEqual("ordinary_control_anchor", assign_evidence_state_v2(off_center, axes, adequacy, calibration)["base_regime"])

    def test_high_evidence_anchor_requires_absolute_semantic_condition_not_quota(self):
        medium_rows = [
            row(pair_id=f"0881/{50 + index}_02", image_a_name=f"image-{50 + index}.jpeg", grayscale_hist_similarity=0.74 + index * 0.01, brightness_gap_abs=7, contrast_gap_abs=4, aspect_ratio_gap_abs=0.04, image_index_gap_abs=2)
            for index in range(6)
        ]
        calibration = fit_calibration_v2(medium_rows)
        top_quota_but_not_semantic_high = medium_rows[-1]
        axes = compute_calibrated_axes_v2(top_quota_but_not_semantic_high, calibration)
        adequacy = compute_adequacy(top_quota_but_not_semantic_high, axes)
        assignment = assign_evidence_state_v2(top_quota_but_not_semantic_high, axes, adequacy, calibration)

        self.assertLess(axes["pair_similarity_axis"], calibration["absolute_thresholds"]["high_similarity"])
        self.assertNotEqual("high_evidence_anchor", assignment["base_regime"])

        semantic_high = row(pair_id="0881/66_02", image_a_name="image-66.jpeg", grayscale_hist_similarity=0.97, brightness_gap_abs=1, contrast_gap_abs=1, aspect_ratio_gap_abs=0.01, image_index_gap_abs=1)
        axes = compute_calibrated_axes_v2(semantic_high, calibration)
        adequacy = compute_adequacy(semantic_high, axes)
        self.assertEqual("high_evidence_anchor", assign_evidence_state_v2(semantic_high, axes, adequacy, calibration)["base_regime"])

    def test_construction_functions_do_not_read_label_or_error_columns(self):
        class PoisonRow(dict):
            def get(self, key, default=None):
                if any(pattern in str(key).lower() for pattern in FORBIDDEN_CONSTRUCTION_PATTERNS):
                    raise AssertionError(f"forbidden key read: {key}")
                return super().get(key, default)

            def __getitem__(self, key):
                if any(pattern in str(key).lower() for pattern in FORBIDDEN_CONSTRUCTION_PATTERNS):
                    raise AssertionError(f"forbidden key read: {key}")
                return super().__getitem__(key)

        poison = PoisonRow(row(final_score=object(), angle_err=object(), gt_distance=object()))
        axes = {
            "observability_axis": compute_observability_axis(poison),
            "pair_similarity_axis": compute_pair_similarity_axis(poison),
            "scale_risk_axis": compute_scale_risk_axis(poison),
            "layout_risk_axis": compute_layout_risk_axis(poison),
        }
        calibration = fit_calibration_v2(TRAIN_ROWS)
        axes = compute_calibrated_axes_v2(poison, calibration)
        adequacy = compute_adequacy(poison, axes)
        assignment = assign_evidence_state_v2(poison, axes, adequacy, calibration)

        self.assertIn(assignment["base_regime"], BASE_REGIMES)


if __name__ == "__main__":
    unittest.main()
