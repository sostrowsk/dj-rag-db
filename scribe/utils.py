import logging
import time
from functools import wraps

import tiktoken

logger = logging.getLogger(__name__)


def tiktoken_length(text: str, model: str = "gpt-5") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found, using cl100k_base encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result

    return wrapper
