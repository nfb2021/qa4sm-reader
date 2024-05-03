from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar('T', bound=Callable[..., Any])


def note(note_text: Any) -> Callable[[T], T]:
    """
    Factory function creating a decorator, that prints a note before the execution of the decorated function.

    Parameters:
    ----------
    note_text : Any
        The note to be printed.

    Returns:
    -------
    Callable[[T], T]
        The decorated function.
    """

    def decorator(func: T) -> T:

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f'\n\n{note_text}\n\n')
            return func(*args, **kwargs)

        return wrapper

    return decorator
