"""Handling optional dependencies."""

try:
    import py3Dmol
except ImportError:
    pass

import sys
from functools import wraps


def requires_extra(module_name, install_name=None):
    """Generate a decorator to require an optional dependency for the given function.

    Parameters
    ----------
    module_name : str
        Name of the module to check for
    install_name : str, optional
        Name of the module to install if it is not found. If not specified, `module_name` is used

    """
    if install_name is None:
        install_name = module_name

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if module_name not in sys.modules:
                raise ImportError(
                    f"{install_name} must be installed to use this function. "
                    f"Install it with `pip install {install_name}` or together with most other optional dependencies with `pip install proteinflow[processing]`."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@requires_extra("py3Dmol")
def _get_view(canvas_size):
    return py3Dmol.view(width=canvas_size[0], height=canvas_size[1])
