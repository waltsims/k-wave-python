"""Utilities for handling deprecation warnings in k-Wave."""

import functools
import warnings
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(message: str, target_version: str = "2.0.0") -> Callable[[F], F]:
    """Decorator to mark functions as deprecated.

    Args:
        message: Message explaining what to use instead
        target_version: Version in which the function will be removed

    Example:
        @deprecated("Use new_function() instead", "2.0.0")
        def old_function():
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in version {target_version}. {message}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
