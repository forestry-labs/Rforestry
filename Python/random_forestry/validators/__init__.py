from typing import Type


def is_positive(t, n) -> bool:
    if not isinstance(t, list):
        return isinstance(n, t) and n > 0
    else:
        return any(isinstance(n, type_curr) for type_curr in t) and n > 0


def is_negative(t: Type, n) -> bool:
    return isinstance(n, t) and n < 0
