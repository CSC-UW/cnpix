"""Tests for the unit-quality tier definitions."""

import pytest

from cnpix.units import thresholds


def test_tiers_are_ordered_permissive_to_strict():
    assert thresholds.QUALITY_TIERS == (
        "mua",
        "sua_permissive",
        "sua_moderate",
        "sua_conservative",
    )


def test_tiers_at_least_includes_stricter_tiers():
    # A unit recorded as sua_conservative also satisfies a sua_moderate request.
    assert thresholds.tiers_at_least("sua_moderate") == [
        "sua_moderate",
        "sua_conservative",
    ]
    assert thresholds.tiers_at_least("mua") == list(thresholds.QUALITY_TIERS)
    assert thresholds.tiers_at_least("sua_conservative") == ["sua_conservative"]


def test_unknown_tier_raises():
    with pytest.raises(ValueError, match="Unknown quality tier"):
        thresholds.tiers_at_least("sua_extremely_moderate")


def test_every_tier_has_threshold_kwargs():
    kwargs = thresholds.get_threshold_kwargs()
    for tier in thresholds.QUALITY_TIERS:
        assert tier in kwargs
    assert "all" in kwargs


def test_sua_moderate_matches_offproj_su1():
    # offproj.units.QUALITY_FILTERS["su1"] is built from these same arguments;
    # the two names must not drift apart.
    assert thresholds.get_threshold_kwargs()["sua_moderate"] == dict(
        required_threshold="conservative",
        isolation_threshold="moderate",
        false_negatives_threshold="moderate",
        presence_threshold=None,
    )
