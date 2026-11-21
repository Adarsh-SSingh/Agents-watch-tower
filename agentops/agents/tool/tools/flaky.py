import time
import multiprocessing


def _execute_flaky_operation(
    mode,
    delay_seconds,
    failure_message,
    call_number,
    result_queue
):
    start_time = time.perf_counter()

    try:
        if mode == "delay":
            time.sleep(delay_seconds)

        elif mode == "error":
            raise RuntimeError(failure_message)

        elif mode == "timeout":
            time.sleep(delay_seconds)

        elif mode == "intermittent":
            if call_number % 2 == 0:
                raise RuntimeError("Simulated intermittent failure.")

        elapsed = round(time.perf_counter() - start_time, 6)

        result_queue.put({
            "status": "success",
            "mode": mode,
            "message": "Flaky tool executed successfully.",
            "execution_time_seconds": elapsed
        })

    except Exception as exc:
        result_queue.put({
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__
        })


# Alternating failure/success state
_intermit_counter = 0


def flaky_tool(
    mode: str = "success",
    delay_seconds: float = 0,
    failure_message: str = "Simulated tool failure",
    timeout_seconds: float = 3
):
    global _intermit_counter

    valid_modes = {
        "success",
        "delay",
        "error",
        "timeout",
        "intermittent"
    }

    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode: {mode}")

    if not isinstance(delay_seconds, (int, float)):
        raise ValueError("Delay must be numeric.")

    if delay_seconds < 0 or delay_seconds > 30:
        raise ValueError("Delay must be between 0 and 30 seconds.")

    if timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("Timeout must be between 0 and 30 seconds.")

    if mode == "intermittent":
        _intermit_counter += 1
        call_number = _intermit_counter
    else:
        call_number = 1

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()

    process = context.Process(
        target=_execute_flaky_operation,
        args=(
            mode,
            delay_seconds,
            failure_message,
            call_number,
            result_queue
        )
    )

    start_time = time.perf_counter()
    process.start()

    # For timeout mode, simulate an operation that exceeds the deadline.
    effective_timeout = (
        timeout_seconds if mode == "timeout"
        else timeout_seconds + 5
    )

    process.join(effective_timeout)

    if process.is_alive():
        process.terminate()
        process.join()

        result_queue.close()

        raise TimeoutError(
            f"Tool execution exceeded {effective_timeout} seconds."
        )

    elapsed = round(time.perf_counter() - start_time, 6)

    try:
        result = result_queue.get(timeout=1)
    except Exception:
        result_queue.close()
        raise RuntimeError("Flaky tool process returned no result.")

    result_queue.close()

    if result["status"] == "error":
        raise RuntimeError(result["error"])

    result["execution_time_seconds"] = elapsed

    return result