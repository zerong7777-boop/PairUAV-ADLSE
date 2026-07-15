from __future__ import annotations

STAGE2_CONTROL_RUNS = [
    {
        "row_id": "E64",
        "name": "e_static_split_bottleneck64",
        "output_mode": "pairuav_msr_e_bottleneck_static_split_heading_range",
        "claim": "compact_bottleneck_control",
        "msr_bottleneck_dim": 64,
        "train_seed": 0,
        "data_seed": 777,
    },
    {
        "row_id": "E256",
        "name": "e_static_split_bottleneck256",
        "output_mode": "pairuav_msr_e_bottleneck_static_split_heading_range",
        "claim": "capacity_monotonicity_control",
        "msr_bottleneck_dim": 256,
        "train_seed": 0,
        "data_seed": 777,
    },
    {
        "row_id": "E128S1",
        "name": "e_static_split_bottleneck128_seed1",
        "output_mode": "pairuav_msr_e_bottleneck_static_split_heading_range",
        "claim": "seed_repeat_control",
        "msr_bottleneck_dim": 128,
        "train_seed": 1,
        "data_seed": 777,
    },
]


def control_ids() -> list[str]:
    return [row["row_id"] for row in STAGE2_CONTROL_RUNS]
