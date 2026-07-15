import torch
import unittest

from reloc3r.loss import PairUAVOfficialMetricAwareLoss


def _view(heading_deg, range_value):
    heading_rad = torch.deg2rad(torch.tensor(heading_deg, dtype=torch.float32))
    return {
        "heading_deg": torch.tensor(heading_deg, dtype=torch.float32),
        "heading_cos": torch.cos(heading_rad),
        "heading_sin": torch.sin(heading_rad),
        "range_value": torch.tensor(range_value, dtype=torch.float32),
    }


def _pred(heading_deg, range_value):
    heading_rad = torch.deg2rad(torch.tensor(heading_deg, dtype=torch.float32))
    return {
        "heading_vec": torch.stack((torch.cos(heading_rad), torch.sin(heading_rad)), dim=-1),
        "range_value": torch.tensor(range_value, dtype=torch.float32),
    }


class PairUAVOfficialMetricAwareLossTest(unittest.TestCase):
    def test_official_loss_zero_for_exact_prediction(self):
        loss_fn = PairUAVOfficialMetricAwareLoss()
        gt2 = _view([30.0, 350.0], [10.0, 20.0])
        pred = _pred([30.0, 350.0], [10.0, 20.0])
        loss, details = loss_fn.compute_loss({}, gt2, {}, pred)
        self.assertLess(float(loss), 1e-6)
        self.assertLess(details["pairuav_official_angle_rel"], 1e-6)
        self.assertLess(details["pairuav_official_distance_rel"], 1e-6)

    def test_official_loss_uses_circular_angle_difference(self):
        loss_fn = PairUAVOfficialMetricAwareLoss(angle_floor_deg=1.0)
        gt2 = _view([359.0], [10.0])
        pred = _pred([1.0], [10.0])
        loss, details = loss_fn.compute_loss({}, gt2, {}, pred)
        self.assertGreater(details["pairuav_official_angle_abs_deg"], 0.0)
        self.assertLess(details["pairuav_official_angle_abs_deg"], 3.0)
        self.assertLess(float(loss), 0.2)

    def test_official_loss_penalizes_relative_distance(self):
        loss_fn = PairUAVOfficialMetricAwareLoss(distance_floor=1.0)
        gt2 = _view([90.0, 90.0], [10.0, 100.0])
        pred = _pred([90.0, 90.0], [20.0, 110.0])
        loss, details = loss_fn.compute_loss({}, gt2, {}, pred)
        self.assertGreater(details["pairuav_official_distance_rel"], 0.0)
        self.assertLess(details["pairuav_official_distance_rel"], 0.6)


if __name__ == "__main__":
    unittest.main()
