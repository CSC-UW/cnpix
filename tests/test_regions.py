"""Tests for Waxholm acronym -> major region mapping."""

from cnpix.units import regions


class TestHippocampusToWaxholm:
    def test_subfield_qualifiers_are_collapsed(self):
        assert regions.hippocampus_to_waxholm("CA1so") == "CA1"
        assert regions.hippocampus_to_waxholm("CA3sr") == "CA3"
        assert regions.hippocampus_to_waxholm("DGsg") == "DG"

    def test_bare_subfields_pass_through(self):
        assert regions.hippocampus_to_waxholm("CA2") == "CA2"

    def test_non_hippocampal_acronyms_are_untouched(self):
        for acronym in ("M2", "PPC", "CLA", "VPM"):
            assert regions.hippocampus_to_waxholm(acronym) == acronym
