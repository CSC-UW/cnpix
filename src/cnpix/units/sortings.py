"""Enumerate and load the CNPIX sortings, without any project-specific gating.

``findlay2025a`` gated its cohort behind a hard-coded ``MANIFEST`` of subjects
that had sharp-wave probes. That is a property of *that* analysis, not of the
dataset, and it silently excluded subjects whose data are perfectly good. Here
the cohort is derived from the sorting registry itself.
"""

from collections import defaultdict
from functools import lru_cache

import wisc_ecephys_tools as wet
from ecephys.units import multi_siks
from ecephys.wne import siutils as wne_siutils
from ecephys.wne.sglx import legacy_sorting
from wisc_ecephys_tools import rats

from cnpix.units.thresholds import get_threshold_kwargs

__all__ = [
    "DEFAULT_EXPERIMENT",
    "get_sortings",
    "get_subjects",
    "load_multiprobe_sorting",
    "load_full_conservative_hypnogram",
]

# As of 2026, sortings exist only for the `full` alias of this experiment;
# `rats.utils.has_sorting` asserts as much.
DEFAULT_EXPERIMENT = rats.constants.SleepDeprivationExperiments.NOD

# The whole-recording, artifact-excluded hypnogram. ACGs are computed over this.
FULL_CONSERVATIVE = "Full.Conservative"


@lru_cache(maxsize=4)
def get_sortings(
    experiment: str = DEFAULT_EXPERIMENT, expand_probes: bool = False
) -> list[tuple[str, str]] | list[tuple[str, tuple[str, ...]]]:
    """Every (subject, probes) with a sorting, anatomy, and a hypnogram.

    Parameters
    ----------
    experiment
        Experiment name.
    expand_probes
        If True, yield one ``(subject, probe)`` per probe. If False (default),
        yield ``(subject, (probe, ...))`` with probes grouped per subject.
    """
    s3 = wet.get_sglx_project("shared")
    tuples = rats.utils.get_subject_experiment_probe_tuples(
        experiment_filter=lambda x: x == experiment
    )
    sortings = [
        (s, p)
        for s, e, p in tuples
        if rats.utils.has_sorting(s, e, p, s3)
        and rats.utils.has_anatomy(s, e, p, s3)
        and rats.utils.has_hypnogram(s, e, None, s3)
    ]
    if expand_probes:
        return sorted(sortings)
    grouped = defaultdict(list)
    for s, p in sortings:
        grouped[s].append(p)
    return sorted((s, tuple(sorted(p))) for s, p in grouped.items())


def get_subjects(experiment: str = DEFAULT_EXPERIMENT) -> list[str]:
    """Subject names with a usable sorting for ``experiment``."""
    return [subject for subject, _ in get_sortings(experiment)]


def load_multiprobe_sorting(
    subject: str,
    experiment: str = DEFAULT_EXPERIMENT,
    quality_tier: str = "mua",
) -> multi_siks.MultiSIKS:
    """Load all of a subject's probes as one sorting, filtered to ``quality_tier``.

    ``quality_tier`` defaults to ``mua`` so that the metric/cell-type tables are
    computed once over the broadest sensible unit set; downstream analyses
    restrict to a stricter tier using the recorded ``max_quality`` column.
    """
    s3 = wet.get_sglx_project("shared")
    probes = dict(get_sortings(experiment))[subject]

    mps = legacy_sorting.load_multiprobe_sorting(
        s3,
        subject,
        experiment,
        probes=list(probes),
        wne_anatomy_project=s3,
    )
    simple_filters, callable_filters = wne_siutils.get_quality_metric_filters(
        **get_threshold_kwargs()[quality_tier]
    )
    return mps.refine_clusters(
        {probe: simple_filters for probe in probes},
        {probe: callable_filters for probe in probes},
        include_nans=True,
    )


def load_full_conservative_hypnogram(
    subject: str, experiment: str = DEFAULT_EXPERIMENT
):
    """The whole-recording conservative hypnogram used to split ACGs by state.

    Uses the probe-consensus hypnogram when a subject has more than one probe,
    matching how the multiprobe sorting pools units across probes.
    """
    s3 = wet.get_sglx_project("shared")
    probes = dict(get_sortings(experiment))[subject]
    probe = None if len(probes) > 1 else probes[0]
    hgs = rats.cnd_hgs.load_statistical_condition_hypnograms(
        subject, experiment, probe, s3
    )
    return hgs[FULL_CONSERVATIVE]
