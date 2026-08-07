"""Unit-quality threshold sets for CNPIX rat analyses.

The dicts returned by :func:`get_threshold_kwargs` are consumable as ``**kwargs``
by :func:`ecephys.wne.siutils.get_quality_metric_filters`.

The tiers are ordered from most permissive to most restrictive:

``all``
    Everything Kilosort/Phy did not label ``noise``.
``mua``
    Passes the required-metric gate (quality label + firing-rate floor) but no
    isolation criteria. Multiunit.
``sua_permissive`` / ``sua_moderate`` / ``sua_conservative``
    Increasingly strict isolation and false-negative criteria.

Note that these are *nested*: a unit passing ``sua_conservative`` also passes
``sua_moderate``, and so on. :func:`cnpix.units.metrics.assign_cluster_quality`
records, per unit, the strictest tier it passes.
"""

__all__ = ["QUALITY_TIERS", "get_threshold_kwargs"]

# Ordered most permissive -> most restrictive. Used both to iterate when
# assigning each unit its strictest passing tier, and to resolve "this tier or
# better" filters downstream.
QUALITY_TIERS: tuple[str, ...] = (
    "mua",
    "sua_permissive",
    "sua_moderate",
    "sua_conservative",
)


def get_threshold_kwargs():
    return dict(
        all=dict(
            required_threshold="all",
            isolation_threshold=None,
            false_negatives_threshold=None,
            presence_threshold=None,
        ),
        mua=dict(
            required_threshold="conservative",
            isolation_threshold=None,
            false_negatives_threshold=None,
            presence_threshold=None,
        ),
        sua_permissive=dict(
            required_threshold="conservative",
            isolation_threshold="permissive",
            false_negatives_threshold="permissive",
            presence_threshold=None,
        ),
        sua_moderate=dict(
            required_threshold="conservative",
            isolation_threshold="moderate",
            false_negatives_threshold="moderate",
            presence_threshold=None,
        ),
        sua_conservative=dict(
            required_threshold="conservative",
            isolation_threshold="conservative",
            false_negatives_threshold="conservative",
            presence_threshold=None,
        ),
    )


def tiers_at_least(tier: str) -> list[str]:
    """Return the quality tiers that are ``tier`` or stricter.

    Useful for filtering a ``max_quality`` column, which records the strictest
    tier each unit passes: a unit whose ``max_quality`` is ``sua_conservative``
    also satisfies a ``sua_moderate`` requirement.
    """
    if tier not in QUALITY_TIERS:
        raise ValueError(f"Unknown quality tier {tier!r}. Expected one of {QUALITY_TIERS}.")
    return list(QUALITY_TIERS[QUALITY_TIERS.index(tier) :])
