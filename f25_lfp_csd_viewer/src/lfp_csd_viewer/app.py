"""CLI entry point for the LFP/CSD Viewer."""

from __future__ import annotations

import logging
import sys

import click
from PySide6.QtWidgets import QApplication

from lfp_csd_viewer.main_window import MainWindow


@click.command()
@click.argument("subject")
@click.argument("experiment")
@click.option(
    "--kind",
    type=click.Choice(["cortical", "hippocampal"]),
    default="cortical",
    help="Which probe region to view.",
)
@click.option(
    "--condition",
    default="Full.Conservative",
    help="Hypnogram condition name for sleep scoring.",
)
@click.option(
    "--page-duration",
    default=10.0,
    type=float,
    help="Page duration in seconds.",
)
@click.option(
    "--gain",
    default=1.0,
    type=float,
    help="Initial LFP gain.",
)
def main(
    subject: str,
    experiment: str,
    kind: str,
    condition: str,
    page_duration: float,
    gain: float,
) -> None:
    """View KCSD heatmaps with LFP trace overlay.

    SUBJECT is the subject name (e.g., CNPIX12-Santiago).
    EXPERIMENT is the experiment name
    (e.g., novel_objects_deprivation).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LFP/CSD Viewer")

    window = MainWindow()
    window._page_duration = page_duration
    window._page_duration_spinbox.setValue(page_duration)
    window._lfp_gain = gain
    window._gain_spinbox.setValue(gain)
    window.show()
    window.load_data(subject, experiment, kind, condition)

    sys.exit(app.exec())
