import concurrent
import dataclasses
from datetime import datetime, timedelta
import functools
import hashlib
import multiprocessing
import os
import random
from typing import Any, Optional, Sequence

import bittensor as bt


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)


def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.
        mode (str): Mode by which the multiprocessing context is obtained. Default to fork for pickling.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


def run_in_thread(func: functools.partial, ttl: int, name=None) -> Any:
    """Runs the provided function on a thread with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(func)
        return future.result(timeout=ttl)
    except concurrent.futures.TimeoutError as e:
        bt.logging.error(f"Failed to complete '{name}' within {ttl} seconds.")
        raise TimeoutError(f"Failed to complete '{name}' within {ttl} seconds.") from e
    finally:
        bt.logging.trace(f"Completed {name}")
        executor.shutdown(wait=False)
        bt.logging.trace(f"{name} cleaned up successfully")


def get_version(filepath: str) -> Optional[int]:
    """Loads a version from the provided filepath or None if the file does not exist.

    Args:
        filepath (str): Path to the version file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            line = f.readline()
            if line:
                return int(line)
            return None
    return None


def save_version(filepath: str, version: int):
    """Saves a version to the provided filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(str(version))


def random_date(start: datetime, end: datetime, seed: int = None) -> datetime:
    """Return a random datetime between two datetimes.

    Args:
        start (datetime): Start of the range, inclusive.
        end (datetime): End of the range, inclusive.
        seed (int): Optional Seed for the random number generator.
    """

    if start.tzinfo != end.tzinfo:
        raise ValueError("Start and end must have the same timezone.")

    if start >= end:
        raise ValueError("Start must be before end.")

    if seed:
        random.seed(seed)

    # Choose a random point between the 2 datetimes.
    random_seconds = random.randint(0, int((end - start).total_seconds()))

    # Add the random seconds to the start time
    return start + timedelta(seconds=random_seconds)

def fingerprint(any: "Sequence[DataclassInstance] | DataclassInstance") -> int:
    """Returns a fingerprint for a Dataclass or sequence of Dataclasses."""
    
    # Convert the dataclass to a string representation of the values
    if isinstance(any, Sequence):
        data_string = str([dataclasses.asdict(x) for x in any]).encode('utf-8')
    else:
        data_string = str(dataclasses.asdict(any)).encode('utf-8')
    return hashlib.sha256(data_string).hexdigest()