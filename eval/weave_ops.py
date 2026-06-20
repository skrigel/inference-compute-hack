from __future__ import annotations

import importlib
import netrc
import os
from collections.abc import Callable
from typing import TypeVar


DEFAULT_WEAVE_PROJECT = "sasha-krigel-massachusetts-institute-of-technology/inference-hack"

F = TypeVar("F", bound=Callable[..., object])


class WeaveSetupError(RuntimeError):
    """Raised when Weave cannot be initialized non-interactively."""


def _load_weave():
    try:
        return importlib.import_module("weave")
    except ModuleNotFoundError as exc:
        raise WeaveSetupError(
            "Weave is not installed. Install it with `pip install weave` and authenticate with W&B."
        ) from exc


def _has_wandb_auth() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return any(credentials.authenticators(host) for host in ("api.wandb.ai", "wandb.ai"))


def init_weave(project: str = DEFAULT_WEAVE_PROJECT):
    """Initialize W&B Weave for eval traces."""

    if not _has_wandb_auth():
        raise WeaveSetupError("W&B auth is missing. Run `wandb login` or set `WANDB_API_KEY` before `--weave`.")
    try:
        return _load_weave().init(project)
    except Exception as exc:
        raise WeaveSetupError(f"Could not initialize Weave project {project!r}: {exc}") from exc


def weave_op(func: F | None = None, *, name: str | None = None):
    """Decorate an eval function when Weave exists; otherwise leave it alone.

    This keeps CI and local mock evals dependency-free while making the same
    function traceable once `weave.init(...)` is enabled on the eval box.
    """

    try:
        weave = importlib.import_module("weave")
    except ModuleNotFoundError:
        weave = None

    def identity(inner: F) -> F:
        return inner

    if weave is None:
        return identity(func) if func is not None else identity

    decorator = weave.op(name=name) if name is not None else weave.op()
    return decorator(func) if func is not None else decorator
