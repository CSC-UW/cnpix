"""Unit-quality threshold sets for CNPIX rat analyses.

The dicts returned by :func:`get_threshold_kwargs` are consumable as ``**kwargs``
by :func:`ecephys.wne.siutils.get_quality_metric_filters`.
"""

__all__ = ["get_threshold_kwargs"]


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
