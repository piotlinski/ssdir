"""Argparse utilities."""
from ast import literal_eval
from typing import Any, Tuple


def parse_kwargs(kwargs_str) -> Tuple[str, Any]:
    """Parse a key-value pair separated by '='."""
    key, value = kwargs_str.split("=")
    return key.strip(), literal_eval(value.strip())
