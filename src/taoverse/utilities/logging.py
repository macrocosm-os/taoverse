import functools
import logging
import sys
import time
from logging import DEBUG, ERROR, INFO, WARNING

from colorama import Back, Fore, Style

# Add a trace level.
_TRACE = 5


class ColorFormatter(logging.Formatter):
    """
    A custom logging formatter derived from the Bittensor logger that applies color to the log message based on the severity level.
    """

    _LEVEL_TO_COLOR = {
        _TRACE: Fore.MAGENTA,
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Back.RED,
    }

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """
        Override format to apply custom color and prefix formatting.
        """
        if record.levelno == _TRACE:
            record.levelname = "TRACE"

        # Generate the timestamp string, including milliseconds.
        created = self.converter(record.created)
        time_str = (
            f'{time.strftime("%Y-%m-%d %H:%M:%S", created)}.{int(record.msecs):03d}'
        )

        # Center the level name and specify the field width of 7 characters to match the longest level name ("WARNING").
        level_str = f"{record.levelname:^7}"
        return logging.Formatter(
            f"{Fore.BLUE}{time_str}{Fore.RESET} | {Style.BRIGHT}{ColorFormatter._LEVEL_TO_COLOR.get(record.levelno, Fore.RESET)}{level_str}{Fore.RESET} | %(message)s",
        ).format(record)


_LOGGER = None
_handler = None


def _initialize_once() -> None:
    global _LOGGER
    global _handler

    if _LOGGER is None:
        _LOGGER = logging.getLogger("taoverse")
        _LOGGER.setLevel(INFO)

        _handler = logging.StreamHandler()
        _handler.setStream(sys.stdout)
        _handler.setFormatter(ColorFormatter())
        _LOGGER.addHandler(_handler)


_initialize_once()

def reinitialize() -> None:
    """Reinitializes the logger and handlers.
    
    Bittensor <= 8.5.0 currently deletes all other loggers handlers so this should be called after the bt logger has been imported / initialized.
    """
    global _LOGGER
    global _handler

    _LOGGER = _LOGGER or logging.getLogger("taoverse")
    _LOGGER.setLevel(INFO)

    for handler in _LOGGER.handlers[:]:
        _LOGGER.removeHandler(handler)
        handler.close()

    _handler = logging.StreamHandler()
    _handler.setStream(sys.stdout)
    _handler.setFormatter(ColorFormatter())
    _LOGGER.addHandler(_handler)


def set_verbosity(verbosity: int) -> None:
    _LOGGER.setLevel(verbosity)


def set_verbosity_trace():
    """Set the verbosity to the `TRACE` level."""
    return set_verbosity(_TRACE)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def trace(message, *args):
    """
    Log a message with level `TRACE` on the root logger.
    """
    _LOGGER.log(_TRACE, message, *args)


def debug(message, *args):
    """
    Log a message with level `DEBUG` on the root logger.
    """
    _LOGGER.debug(message, *args)


def info(message, *args):
    """
    Log a message with level `INFO` on the root logger.
    """
    _LOGGER.info(message, *args)


def warning(message, *args):
    """
    Log a message with level `WARNING` on the root logger.
    """
    _LOGGER.warning(message, *args)


def error(message, *args):
    """
    Log a message with level `ERROR` on the root logger.
    """
    _LOGGER.error(message, *args)


@functools.lru_cache(None)
def trace_once(message, *args):
    """
    This method is identical to `logger.trace()`, but will emit the trace with the same message only once
    """
    trace(message, *args)


@functools.lru_cache(None)
def debug_once(message, *args):
    """
    This method is identical to `logger.debug()`, but will emit the debug with the same message only once
    """
    debug(message, *args)


@functools.lru_cache(None)
def info_once(message, *args):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once
    """
    info(message, *args)


@functools.lru_cache(None)
def warning_once(message, *args):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once
    """
    warning(message, *args)


@functools.lru_cache(None)
def error_once(message, *args):
    """
    This method is identical to `logger.error()`, but will emit the error with the same message only once
    """
    error(message, *args)
