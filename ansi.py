from enum import Enum
from typing import Iterable, Union


class TEXT(Enum):
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37

    BOLD = 1
    UNDER = 4
    INVERT = 7


class BACK(Enum):
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLOW = 43
    BLUE = 44
    MAGENTA = 45
    CYAN = 46
    WHITE = 47


def modify(s: str, codes: Iterable[Union[TEXT, BACK]]) -> str:
    codes_str = [str(code.value) for code in codes]
    return '\033[' + ';'.join(codes_str) + 'm' + s + '\033[0m'
