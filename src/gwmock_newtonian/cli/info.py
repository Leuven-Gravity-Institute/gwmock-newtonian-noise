# ruff: noqa PL0415
"""Info command module."""

from __future__ import annotations

import typer


def info_command() -> None:
    """Print package version and public joint-backend discovery status."""
    from logging import getLogger

    from gwmock_noise import available_joint_backend_names

    from gwmock_newtonian import __version__
    from gwmock_newtonian.simulator import BACKEND_NAME

    logger = getLogger("gwmock_newtonian")

    typer.echo(f"gwmock_newtonian {__version__}")
    backends = available_joint_backend_names()
    registered = BACKEND_NAME in backends
    typer.echo(f"gwmock_noise.joint_backends discovered: {', '.join(backends) or '(none)'}")
    typer.echo(f"{BACKEND_NAME} registered: {registered}")

    logger.info("Executed info_command; %d joint backend(s) discovered", len(backends))
