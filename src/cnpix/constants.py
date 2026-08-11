"""Constants shared across every ``cnpix`` area.

Anything here is a property of the CNPIX dataset itself, not of a particular
analysis, so all subpackages read it from one place rather than redefining it.
"""

from __future__ import annotations

from wisc_ecephys_tools import rats

__all__ = ["DEFAULT_EXPERIMENT"]

#: The experiment nearly all CNPIX work targets.
DEFAULT_EXPERIMENT: str = rats.constants.SleepDeprivationExperiments.NOD
