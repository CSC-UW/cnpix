"""Shared configuration for OFF-label evaluation against manual ground truth.

Defines the two evaluation conditions (NREM / Wake) and the per-condition
image-selection convention.

Image-selection convention (the crux of comparing against partial ground truth):

- **NREM** (``Early.REC.NREM``): manual labels cover only a small fraction of
  images; an unlabeled image was *not inspected*, NOT confirmed OFF-free. So we
  evaluate only on chunks that contain at least one manual label
  (``chunks="labeled"``).
- **Wake** (``Late.NOD.Wake``): every image was inspected, so an unlabeled image
  is a true negative. We evaluate on all chunks (``chunks="all"``).
"""

from __future__ import annotations

NREM_CONDITION = "Early.REC.NREM"
WAKE_CONDITION = "Late.NOD.Wake"

# The Wake image stacks + timestamps.zarr were written under a TRUNCATED
# condition directory ("Late.NOD"), while manual labels and the morphological
# parquet use the canonical "Late.NOD.Wake". Resolve eval condition -> stack dir.
STACK_CONDITION = {WAKE_CONDITION: "Late.NOD"}


def stack_condition(condition: str) -> str:
    """Map an evaluation condition to its on-disk image-stack directory name."""
    return STACK_CONDITION.get(condition, condition)


# Filters: all three for NREM; only llas is defined for Wake (clas/blas are
# sleep-only). Which trained model a given evaluation is scored against is a
# SAM3 concern and lives in ``samoffs.config.MODEL_KEYS``.
EVAL_CONFIGS: dict[str, dict] = {
    "NREM": {
        "condition": NREM_CONDITION,
        "chunks": "labeled",
        "filters": ("llas", "clas", "blas"),
    },
    "Wake": {
        "condition": WAKE_CONDITION,
        "chunks": "all",
        "filters": ("llas",),
    },
}
