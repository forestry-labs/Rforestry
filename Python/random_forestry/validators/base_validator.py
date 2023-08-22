import functools
from abc import ABC, abstractmethod


class BaseValidator(ABC):
    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, function)

    def __get__(self, obj, _):
        return functools.partial(self.__call__, obj)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...
