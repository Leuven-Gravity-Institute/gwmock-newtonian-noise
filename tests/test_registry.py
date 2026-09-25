"""Tests for the entry-point plugin registration under gwmock_noise.joint_backends.

Also demonstrates the "missing provider fails clearly" acceptance criterion:
this package's own test suite necessarily has gwmock-newtonian installed (it
tests itself), so a literal uninstalled-provider environment is not
reproducible as a permanent, fast unit test here -- it would need a network
install into a throwaway venv on every run. What is verified below instead:
(1) when installed, this backend is discoverable through gwmock_noise's own
public discovery API with no private import; (2) gwmock_noise's discovery
mechanism raises a clear, listing KeyError for any name it does not find
registered -- which is exactly what a public environment without this
package sees when it asks for "newtonian_scaffold" by name.

This was additionally checked, once, by hand in a genuinely provider-absent
environment (a fresh venv with only gwmock-noise installed, pinned to the
same commit, and gwmock-newtonian never installed into it):

    available backends: ('dummy_correlated',)
    KeyError raised as expected: "Unknown joint backend 'newtonian_scaffold'.
    Available: dummy_correlated."

That confirms the stand-in below reproduces the real failure shape, but it
was a one-off manual check in a throwaway venv, not something this suite
re-runs automatically.
"""

from __future__ import annotations

import pytest
from gwmock_noise import available_joint_backend_names, load_joint_backend

from gwmock_newtonian.simulator import BACKEND_NAME, NewtonianScaffoldSimulator

pytestmark = pytest.mark.unit


def test_backend_name_is_discoverable_with_no_private_import() -> None:
    """The public discovery surface finds this backend purely through package metadata."""
    assert BACKEND_NAME in available_joint_backend_names()


def test_backend_loads_to_the_expected_class() -> None:
    """Loading the registered entry point resolves to this package's simulator class."""
    loaded = load_joint_backend(BACKEND_NAME)
    assert loaded is NewtonianScaffoldSimulator


def test_missing_provider_fails_clearly() -> None:
    """Requesting an unregistered backend name raises a clear, listing KeyError.

    This is the failure mode a public environment without gwmock-newtonian
    installed sees: the entry point is simply absent from discovery, and
    load_joint_backend raises this same KeyError shape.
    """
    with pytest.raises(KeyError, match="Unknown joint backend"):
        load_joint_backend("not_a_registered_backend")
