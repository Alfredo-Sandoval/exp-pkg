"""Exception translation scopes.

This module provides small context managers that translate exceptions without
`try/except` at the call site. Use these at IO/process boundaries to surface
clean error messages while keeping the core codepath fail-fast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType


@dataclass(frozen=True, slots=True)
class ExceptionTranslationRule:
    """A single exception-translation rule.

    If an exception is raised inside the managed scope and its type is a subclass
    of ``catch``, it is replaced by ``raise_type(message)`` with the original
    exception preserved as the cause.
    """

    catch: tuple[type[BaseException], ...]
    raise_type: type[Exception]
    message: str | Callable[[BaseException], str]


class TranslateExceptions:
    """Translate exceptions raised inside the scope via ``__exit__``.

    Example:
        with translate_exceptions(
            ExceptionTranslationRule(
                catch=(OSError,),
                raise_type=RuntimeError,
                message="Failed to read file",
            )
        ):
            ...
    """

    def __init__(self, *rules: ExceptionTranslationRule) -> None:
        self._rules = rules

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None or exc is None:
            return False

        for rule in self._rules:
            if issubclass(exc_type, rule.catch):
                message = rule.message if isinstance(rule.message, str) else rule.message(exc)
                raise rule.raise_type(message) from exc

        return False


def translate_exceptions(*rules: ExceptionTranslationRule) -> TranslateExceptions:
    """Return a context manager that translates exceptions according to ``rules``.

    Each rule maps a set of caught exception types to a new exception type and
    message. When multiple rules match, the first matching rule wins.

    Args:
        *rules: One or more :class:`ExceptionTranslationRule` instances.

    Returns:
        A :class:`TranslateExceptions` context manager.
    """
    return TranslateExceptions(*rules)


def raise_as(
    raise_type: type[Exception],
    *,
    catch: tuple[type[BaseException], ...],
    message: str | Callable[[BaseException], str],
) -> TranslateExceptions:
    """Shorthand for a single-rule :func:`translate_exceptions` scope.

    Equivalent to::

        translate_exceptions(ExceptionTranslationRule(catch=catch, raise_type=raise_type, message=message))

    Args:
        raise_type: The exception type to raise when a caught exception matches.
        catch: Tuple of exception types to intercept inside the scope.
        message: Static string or callable ``(exc) -> str`` used to build the
            new exception message. The original exception is preserved as the
            cause via ``raise ... from exc``.

    Returns:
        A :class:`TranslateExceptions` context manager.
    """
    return translate_exceptions(
        ExceptionTranslationRule(catch=catch, raise_type=raise_type, message=message)
    )


class CaptureExceptions:
    """Capture and suppress expected exceptions raised inside the scope."""

    def __init__(self, catch: tuple[type[BaseException], ...]) -> None:
        self._catch = catch
        self.exc: BaseException | None = None
        self.tb: TracebackType | None = None

    def __enter__(self) -> CaptureExceptions:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None or exc is None:
            return False
        if not issubclass(exc_type, self._catch):
            return False
        self.exc = exc
        self.tb = tb
        return True


class OnException:
    """Run a callback when an expected exception occurs (then re-raise)."""

    def __init__(
        self,
        *,
        catch: tuple[type[BaseException], ...],
        handler: Callable[[type[BaseException], BaseException, TracebackType | None], None],
    ) -> None:
        self._catch = catch
        self._handler = handler

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None or exc is None:
            return False
        if not issubclass(exc_type, self._catch):
            return False
        self._handler(exc_type, exc, tb)
        return False


def on_exception(
    *,
    catch: tuple[type[BaseException], ...],
    handler: Callable[[type[BaseException], BaseException, TracebackType | None], None],
) -> OnException:
    """Return a context manager that calls ``handler`` when a matching exception occurs.

    Unlike :func:`translate_exceptions`, this does **not** suppress or replace
    the exception — it re-raises it after invoking the handler. Use this for
    side-effects such as logging or cleanup that must run before the exception
    propagates.

    Args:
        catch: Tuple of exception types to observe inside the scope.
        handler: Callable invoked as ``handler(exc_type, exc, tb)`` when a
            matching exception is raised.

    Returns:
        An :class:`OnException` context manager.
    """
    return OnException(catch=catch, handler=handler)


__all__ = [
    "CaptureExceptions",
    "ExceptionTranslationRule",
    "OnException",
    "TranslateExceptions",
    "on_exception",
    "raise_as",
    "translate_exceptions",
]
