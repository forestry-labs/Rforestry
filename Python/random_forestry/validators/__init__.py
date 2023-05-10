def positive_integer(n):
    return isinstance(n, int) and n > 0


def positive_float(n):
    return isinstance(n, float) and n > 0


def negative_integer(n):
    return isinstance(n, int) and n < 0


def negative_float(n):
    return isinstance(n, float) and n < 0
