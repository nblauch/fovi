"""Validate optional capability dependencies without importing their runtimes."""

from importlib.util import find_spec


def require_dependencies(extra: str, modules: tuple[str, ...]) -> None:
    """Raise with an installation command if a capability is unavailable."""
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            f"Missing dependencies for fovi[{extra}]: {', '.join(missing)}. "
            f"Install with `pip install 'fovi[{extra}]'`. "
            "See the installation guide for native FFCV prerequisites when using training."
        )
