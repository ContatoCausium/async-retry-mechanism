import asyncio

import pytest

from async_retry.retry import retry_async


@pytest.mark.asyncio
async def test_returns_result_on_first_attempt() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    assert await retry_async(operation, sleep=fake_sleep) == "ok"
    assert calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_retries_transient_failures_with_exponential_backoff() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("temporary")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    assert await retry_async(
        operation,
        max_attempts=4,
        initial_delay=0.25,
        multiplier=2.0,
        max_delay=1.0,
        retry_on=(TimeoutError,),
        sleep=fake_sleep,
    ) == "ok"
    assert calls == 3
    assert delays == [0.25, 0.5]


@pytest.mark.asyncio
async def test_raises_final_retryable_error_without_final_sleep() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("temporary")

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(TimeoutError, match="temporary"):
        await retry_async(
            operation,
            max_attempts=3,
            initial_delay=1.0,
            multiplier=2.0,
            retry_on=(TimeoutError,),
            sleep=fake_sleep,
        )

    assert calls == 3
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_does_not_retry_non_recoverable_error() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("invalid input")

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(ValueError, match="invalid input"):
        await retry_async(operation, retry_on=(TimeoutError,), sleep=fake_sleep)

    assert calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_caps_each_backoff_delay() -> None:
    delays: list[float] = []

    async def operation() -> None:
        raise TimeoutError("temporary")

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    with pytest.raises(TimeoutError):
        await retry_async(
            operation,
            max_attempts=4,
            initial_delay=5.0,
            multiplier=2.0,
            max_delay=1.0,
            retry_on=(TimeoutError,),
            sleep=fake_sleep,
        )

    assert delays == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_propagates_cancellation_from_operation() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_async(operation, max_attempts=3, retry_on=(TimeoutError,))

    assert calls == 1


@pytest.mark.asyncio
async def test_propagates_cancellation_from_sleep() -> None:
    calls = 0

    async def operation() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("temporary")

    async def cancelled_sleep(_: float) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await retry_async(
            operation,
            max_attempts=3,
            retry_on=(TimeoutError,),
            sleep=cancelled_sleep,
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_retries_with_zero_delay() -> None:
    calls = 0
    delays: list[float] = []

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary")
        return "ok"

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    assert await retry_async(
        operation,
        initial_delay=0,
        multiplier=2,
        max_delay=0,
        retry_on=(TimeoutError,),
        sleep=fake_sleep,
    ) == "ok"
    assert delays == [0.0]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_attempts": 0}, ValueError),
        ({"max_attempts": True}, TypeError),
        ({"initial_delay": -1}, ValueError),
        ({"max_delay": float("inf")}, ValueError),
        ({"multiplier": 0.5}, ValueError),
        ({"retry_on": ()}, TypeError),
        ({"retry_on": (BaseException,)}, TypeError),
    ],
)
@pytest.mark.asyncio
async def test_validates_configuration(kwargs: dict[str, object], error: type[Exception]) -> None:
    async def operation() -> None:
        return None

    with pytest.raises(error):
        await retry_async(operation, **kwargs)  # type: ignore[arg-type]
