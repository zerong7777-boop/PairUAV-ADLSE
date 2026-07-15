from reloc3r.loss import *  # noqa: F401,F403


EXPR = (
    "__import__('phase104i_tail_loss')."
    "PairUAVTailWeightedOfficialLoss("
    "heading_weight=1.0, range_weight=1.0, angle_floor_deg=1.0, "
    "distance_floor=1.0, absolute_heading_weight=0.05, absolute_range_weight=0.10, "
    "tail_start=80.0, tail_end=120.0, tail_max_weight=3.0)"
)


if __name__ == "__main__":
    criterion = eval(EXPR)
    print(type(criterion).__name__)
    print(criterion.tail_start, criterion.tail_end, criterion.tail_max_weight, criterion.absolute_range_weight)

    import torch

    heading_deg = torch.tensor([0.0, 120.0, -160.0])
    heading_rad = torch.deg2rad(heading_deg)
    pose2 = {
        "heading_vec": torch.stack([torch.cos(heading_rad), torch.sin(heading_rad)], dim=-1),
        "range_value": torch.tensor([[0.5], [80.0], [66.0]]),
    }
    gt2 = {
        "heading_deg": torch.tensor([0.0, 100.0, -160.0]),
        "range_value": torch.tensor([0.0, 122.5, 110.0]),
    }
    loss, details = criterion.compute_loss({}, gt2, {}, pose2)
    print(f"loss={float(loss):.6f}")
    for key in sorted(details):
        print(f"{key}={details[key]}")
