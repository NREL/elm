# -*- coding: utf-8 -*-
# fmt: off
"""ELM Ordinances CLI."""
import sys
import json
import click
import asyncio
import logging

from elm.version import __version__
from elm.ords.process import process_counties_with_openai


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    """ELM ordinances command line interface."""
    ctx.ensure_object(dict)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to ordinance configuration JSON file. This file "
                   "should contain any/all the arguments to pass to "
                   ":func:`elm.ords.process.process_counties_with_openai`.")
@click.option("-v", "--verbose", is_flag=True,
              help="Flag to show logging on the terminal. Default is not "
                   "to show any logs on the terminal.")
def ords(config, verbose):
    """Download and extract ordinances for a list of counties."""
    with open(config, "r") as fh:
        config = json.load(fh)

    if verbose:
        logger = logging.getLogger("elm")
        logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        logger.setLevel(config.get("log_level", "INFO"))

    # asyncio.run(...) doesn't throw exceptions correctly for some reason...
    loop = asyncio.get_event_loop()
    loop.run_until_complete(process_counties_with_openai(**config))


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main(obj={})
