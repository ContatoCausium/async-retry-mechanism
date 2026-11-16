"""Asynchronous retry support with bounded exponential backoff."""

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
Sleep = Callable[[float], Awaitable[None]]


def _finite_non_negative(name: str, value: float) -> float:
    """Validate and return a finite, non-negative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")

    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    multiplier: float = 2.0,
    max_delay: float = 5.0,
    retry_on: tuple[type[Exception], ...] = (TimeoutError,),
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Run an async operation with bounded retries and exponential backoff."""
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not callable(sleep):
        raise TypeError("sleep must be callable")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    initial = _finite_non_negative("initial_delay", initial_delay)
    cap = _finite_non_negative("max_delay", max_delay)
    factor = _finite_non_negative("multiplier", multiplier)
    if factor < 1:
        raise ValueError("multiplier must be greater than or equal to 1")

    if not isinstance(retry_on, tuple) or not retry_on:
        raise TypeError("retry_on must be a non-empty tuple of Exception types")
    if any(
        not isinstance(exception_type, type)
        or not issubclass(exception_type, Exception)
        for exception_type in retry_on
    ):
        raise TypeError("retry_on must contain only Exception subclasses")

    delay = min(initial, cap)

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except retry_on:
            if attempt == max_attempts:
                raise

        await sleep(delay)
        delay = min(delay * factor, cap)

    raise AssertionError("Unreachable")
