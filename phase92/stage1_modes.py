from __future__ import annotations

STAGE1_RUNS = [
    {
        "row_id": "C",
        "name": "two_bottlenecks_all_evidence",
        "output_mode": "pairuav_msr_c_two_bottleneck_heading_range",
        "claim": "factorized_relation_variables",
    },
    {
        "row_id": "D",
        "name": "static_evidence_split",
        "output_mode": "pairuav_msr_d_static_split_heading_range",
        "claim": "static_factor_evidence_split",
    },
    {
        "row_id": "E",
        "name": "two_bottlenecks_static_split",
        "output_mode": "pairuav_msr_e_bottleneck_static_split_heading_range",
        "claim": "both_factorization_and_evidence_split",
    },
]


def output_modes() -> list[str]:
    return [row["output_mode"] for row in STAGE1_RUNS]

