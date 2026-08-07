"""``cnpix-units`` command line interface."""

import logging

import click

from cnpix.units import pipeline, sortings

_SUBJECT_HELP = "Subject name. Repeatable. Defaults to every subject with a sorting."


def _resolve(subject: tuple[str, ...], experiment: str) -> list[str]:
    return list(subject) if subject else sortings.get_subjects(experiment)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Log progress.")
def main(verbose: bool) -> None:
    """Unit metrics and cell-type classification for the CNPIX dataset."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )


_experiment_option = click.option(
    "--experiment", default=sortings.DEFAULT_EXPERIMENT, show_default=True
)
_subject_option = click.option("-s", "--subject", multiple=True, help=_SUBJECT_HELP)


@main.command("list-subjects")
@_experiment_option
def list_subjects(experiment: str) -> None:
    """Print every subject with a sorting, anatomy, and a hypnogram."""
    for subject, probes in sortings.get_sortings(experiment):
        click.echo(f"{subject}\t{','.join(probes)}")


@main.command("acgs")
@_subject_option
@_experiment_option
@click.option("--overwrite", is_flag=True, help="Recompute even if ACGs exist.")
def acgs(subject: tuple[str, ...], experiment: str, overwrite: bool) -> None:
    """Step 1: compute narrow and wide autocorrelograms."""
    for name in _resolve(subject, experiment):
        click.echo(f"ACGs: {name}")
        pipeline.compute_subject_acgs(name, experiment, overwrite=overwrite)


@main.command("acg-metrics")
@_subject_option
@_experiment_option
def acg_metrics(subject: tuple[str, ...], experiment: str) -> None:
    """Step 2: fit ACGs and derive per-unit metrics."""
    for name in _resolve(subject, experiment):
        click.echo(f"ACG metrics: {name}")
        pipeline.compute_subject_acg_metrics(name, experiment)


@main.command("cohort-tables")
@_subject_option
@_experiment_option
def cohort_tables(subject: tuple[str, ...], experiment: str) -> None:
    """Step 3: join metrics with sorting properties and assign quality tiers."""
    pipeline.build_cohort_tables(_resolve(subject, experiment), experiment)


@main.command("cell-types")
@_experiment_option
def cell_types(experiment: str) -> None:
    """Step 4: classify putative cell types."""
    pipeline.assign_cohort_cell_types(experiment)


@main.command("run-all")
@_subject_option
@_experiment_option
@click.option("--overwrite", is_flag=True, help="Recompute ACGs even if they exist.")
def run_all(subject: tuple[str, ...], experiment: str, overwrite: bool) -> None:
    """Run every step in order."""
    pipeline.run_all(_resolve(subject, experiment), experiment, overwrite=overwrite)


if __name__ == "__main__":
    main()
