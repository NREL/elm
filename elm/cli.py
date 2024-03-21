# -*- coding: utf-8 -*-
"""ELM Ordinances CLI."""
import json
import click
import asyncio

from elm.version import __version__
from elm.ords.process import process_counties_with_openai


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    """ELM ordinances command line interface."""
    ctx.ensure_object(dict)


@main.group()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Path to ordinance configuration JSON file. This file "
        "should contain all the arguments to pass to "
        "elm.ords.process.process_counties_with_openai"
    ),
)
def ords(config):
    """Download and extract ordinances for a list of counties."""
    with open(config, "r") as fh:
        config = json.load(fh)

    asyncio.run(process_counties_with_openai(**config))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main(obj={})
