import unittest

from scripts.phase27_a_evidence_state_v3_calibration import (
    AXIS_NAMES,
    BAND_VALUES,
    BASE_REGIMES,
    CALIBRATION_VERSION,
    FORBIDDEN_CONSTRUCTION_PATTERNS,
    audit_calibration_columns,
    assign_calibrated_evidence_state,
    compute_calibrated_axes,
    fit_calibration,
    rank01,
    safe_float,
)


BASE_ROW = {
    "source_split": "train",
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
}


def row(**overrides):
    out = dict(BASE_ROW)
    out.update(overrides)
    return out


TRAIN_ROWS = [
    row(pair_id="low", brightness_gap_abs=38, contrast_gap_abs=24, grayscale_hist_similarity=0.22, aspect_ratio_gap_abs=0.42, image_index_gap_abs=17),
    row(pair_id="ordinary_a", brightness_gap_abs=12, contrast_gap_abs=8, grayscale_hist_similarity=0.58, aspect_ratio_gap_abs=0.10, image_index_gap_abs=5),
    row(pair_id="ordinary_b", brightness_gap_abs=8, contrast_gap_abs=4, grayscale_hist_similarity=0.72, aspect_ratio_gap_abs=0.06, image_index_gap_abs=3),
    row(pair_id="high", brightness_gap_abs=1, contrast_gap_abs=1, grayscale_hist_similarity=0.96, aspect_ratio_gap_abs=0.01, image_index_gap_abs=1),
]


class CalibrationCoreTests(unittest.TestCase):
    def test_audit_rejects_forbidden_construction_columns(self):
        for pattern in FORBIDDEN_CONSTRUCTION_PATTERNS:
            audit = audit_calibration_columns(["pair_id", f"prefix_{pattern}_suffix"])
            self.assertFalse(audit["passed"], pattern)
            self.assertEqual([f"prefix_{pattern}_suffix"], audit["forbidden_columns"])

        with self.assertRaises(ValueError):
            fit_calibration([row(heading_num=12)])

    def test_fit_calibration_uses_train_split_and_feature_columns_only(self):
        rows = TRAIN_ROWS + [
            row(source_split="dev", brightness_gap_abs=999, contrast_gap_abs=999, grayscale_hist_similarity=0.99),
        ]
        calibration = fit_calibration(rows)

        self.assertEqual(CALIBRATION_VERSION, calibration["calibration_version"])
        self.assertEqual("source_split_train", calibration["fit_scope"])
        self.assertEqual(4, calibration["fit_row_count"])
        for axis in AXIS_NAMES:
            self.assertIn(axis, calibration["axis_thresholds"])
        used_columns = set(calibration["feature_columns_used"])
        self.assertIn("brightness_gap_abs", used_columns)
        self.assertIn("grayscale_hist_similarity", used_columns)
        self.assertNotIn("source_split", used_columns)
        self.assertTrue(used_columns.isdisjoint(FORBIDDEN_CONSTRUCTION_PATTERNS))

    def test_fit_calibration_records_all_rows_scope_when_no_split(self):
        rows = [{k: v for k, v in item.items() if k != "source_split"} for item in TRAIN_ROWS]
        calibration = fit_calibration(rows)
        self.assertEqual("all_rows_no_split", calibration["fit_scope"])
        self.assertEqual(len(rows), calibration["fit_row_count"])

    def test_fit_calibration_records_all_rows_scope_when_split_has_no_train_rows(self):
        rows = [row(source_split="dev", pair_id=f"dev_{index}") for index, _item in enumerate(TRAIN_ROWS)]
        calibration = fit_calibration(rows)
        self.assertEqual("all_rows_no_split", calibration["fit_scope"])
        self.assertEqual(len(rows), calibration["fit_row_count"])

    def test_compute_axes_emits_expected_axes_and_valid_bands(self):
        calibration = fit_calibration(TRAIN_ROWS)
        axes = compute_calibrated_axes(row(), calibration)

        for axis in AXIS_NAMES:
            self.assertIn(axis, axes)
            self.assertIn(f"{axis}_band", axes)
            self.assertIn(axes[f"{axis}_band"], BAND_VALUES)
            self.assertIsInstance(axes[axis], float)

    def test_assignment_emits_exactly_one_base_regime(self):
        calibration = fit_calibration(TRAIN_ROWS)
        axes = compute_calibrated_axes(row(), calibration)
        assignment = assign_calibrated_evidence_state(row(), axes, calibration)

        self.assertIn(assignment["base_regime"], BASE_REGIMES)
        flags = [assignment[f"base_{regime}"] for regime in BASE_REGIMES]
        self.assertEqual(1, sum(flags))
        self.assertEqual(CALIBRATION_VERSION, assignment["calibration_version"])
        self.assertEqual(calibration["fit_scope"], assignment["calibration_source"])
        for axis in AXIS_NAMES:
            self.assertIn(axis, assignment)
            self.assertIn(f"{axis}_band", assignment)

    def test_ordinary_control_anchor_requires_mid_axes_and_non_missing_cheap_features(self):
        calibration = fit_calibration(TRAIN_ROWS)
        ordinary = row(brightness_gap_abs=12, contrast_gap_abs=8, grayscale_hist_similarity=0.58, aspect_ratio_gap_abs=0.10, image_index_gap_abs=5)
        axes = compute_calibrated_axes(ordinary, calibration)
        self.assertEqual("mid", axes["observability_axis_band"])
        self.assertEqual("mid", axes["pair_similarity_axis_band"])
        assignment = assign_calibrated_evidence_state(ordinary, axes, calibration)
        self.assertEqual("ordinary_control_anchor", assignment["base_regime"])

        missing_cheap = row(has_cheap_image_features=0)
        axes = compute_calibrated_axes(missing_cheap, calibration)
        assignment = assign_calibrated_evidence_state(missing_cheap, axes, calibration)
        self.assertEqual("unknown_insufficient_features", assignment["base_regime"])

    def test_high_evidence_anchor_not_from_high_similarity_alone_when_risk_high(self):
        calibration = fit_calibration(TRAIN_ROWS)
        risky = row(brightness_gap_abs=1, contrast_gap_abs=1, grayscale_hist_similarity=0.99, aspect_ratio_gap_abs=0.55, image_index_gap_abs=1)
        axes = compute_calibrated_axes(risky, calibration)
        self.assertEqual("high", axes["pair_similarity_axis_band"])
        self.assertEqual("high", axes["scale_compatibility_axis_band"])
        assignment = assign_calibrated_evidence_state(risky, axes, calibration)
        self.assertNotEqual("high_evidence_anchor", assignment["base_regime"])
        self.assertEqual("ambiguous_unreliable", assignment["base_regime"])

    def test_cached_matcher_fields_are_optional_feature_anchors_not_targets(self):
        rows = TRAIN_ROWS + [row(cached_match_count=120, cached_scale_balance=0.98, cached_spatial_entropy=0.91, cached_anchor_spread=0.85)]
        calibration = fit_calibration(rows)
        self.assertIn("cached_match_count", calibration["feature_columns_used"])
        self.assertTrue(set(calibration["feature_columns_used"]).isdisjoint(FORBIDDEN_CONSTRUCTION_PATTERNS))
        self.assertEqual([], calibration["target_columns_used"])

        with_cached = row(cached_match_count=120, cached_scale_balance=0.99, cached_spatial_entropy=0.9, cached_anchor_spread=0.9)
        without_cached = row()
        axes_cached = compute_calibrated_axes(with_cached, calibration)
        axes_plain = compute_calibrated_axes(without_cached, calibration)
        self.assertGreaterEqual(axes_cached["pair_similarity_axis"], axes_plain["pair_similarity_axis"])
        self.assertLessEqual(axes_cached["scale_compatibility_axis"], axes_plain["scale_compatibility_axis"])

    def test_helpers_handle_numeric_edge_cases(self):
        self.assertIsNone(safe_float("nan"))
        self.assertEqual(3.5, safe_float("3.5"))
        self.assertEqual([0.0, None, 0.5, 1.0], rank01([2, None, 4, 6]))


if __name__ == "__main__":
    unittest.main()
