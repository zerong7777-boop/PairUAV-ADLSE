import torch

from train import loss_of_one_batch


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, view1, view2):
        self.calls.append((view1["name"], view2["name"]))
        batch = view1["img"].shape[0]
        pose = {
            "heading_vec": torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
            "range_value": torch.zeros(batch, 1, dtype=torch.float32),
        }
        return pose, pose


class NeedsSwappedCriterion(torch.nn.Module):
    requires_swapped_forward = True

    def forward(self, gt1, gt2, pose1, pose2, **kw):
        assert "swapped_pose1" in kw
        assert "swapped_pose2" in kw
        return torch.tensor(0.0, requires_grad=True), {"ok": 1.0}


class PlainCriterion(torch.nn.Module):
    def forward(self, gt1, gt2, pose1, pose2, **kw):
        assert "swapped_pose1" not in kw
        assert "swapped_pose2" not in kw
        return torch.tensor(0.0, requires_grad=True), {"ok": 1.0}


def _view(name):
    return {
        "name": name,
        "img": torch.zeros(1, 3, 4, 4),
        "heading_deg": torch.tensor([0.0]),
        "range_value": torch.tensor([0.0]),
    }


def test_loss_of_one_batch_runs_second_forward_when_criterion_requires_it():
    model = CountingModel()
    batch = (_view("A"), _view("B"))

    loss, details = loss_of_one_batch(
        batch,
        model,
        NeedsSwappedCriterion(),
        torch.device("cpu"),
        use_amp=False,
        ret="loss",
    )

    assert float(loss) == 0.0
    assert details["ok"] == 1.0
    assert model.calls == [("A", "B"), ("B", "A")]


def test_loss_of_one_batch_keeps_single_forward_for_plain_criterion():
    model = CountingModel()
    batch = (_view("A"), _view("B"))

    loss_of_one_batch(
        batch,
        model,
        PlainCriterion(),
        torch.device("cpu"),
        use_amp=False,
        ret="loss",
    )

    assert model.calls == [("A", "B")]
