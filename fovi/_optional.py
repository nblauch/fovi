"""Validate optional capability dependencies without importing their runtimes."""

from importlib.util import find_spec


def require_dependencies(extra: str, modules: tuple[str, ...]) -> None:
    """Raise with an installation command if a capability is unavailable."""
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        raise ModuleNotFoundError(
            f"Missing dependencies for fovi[{extra}]: {', '.join(missing)}. "
            f"Install with `pip install 'fovi[{extra}]'`."
        )


def require_ffcv() -> None:
    """Require the externally installed FFCV-SSL runtime for built-in loaders."""
    if find_spec("ffcv") is None:
        raise ModuleNotFoundError(
            "The built-in data loaders require FFCV-SSL. Install FFCV-SSL manually "
            "with its native prerequisites; see "
            "https://github.com/nblauch/fovi#manual-ffcv-installation. "
            "Fovi extras do not install FFCV."
        )
