# This is a workaround for relative imports to work in this file
# # Add the parent directory of the current script's directory to the Python path
# import sys
# import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
# sys.path.append(os.path.dirname(SCRIPT_DIR))  # Add the parent directory to the Python path

import pytest


def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if not isinstance(n, int):
        raise TypeError("Factorial is not defined for non-integer numbers.")
    elif n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)

@pytest.mark.parametrize("input_number, expected_factorial", [
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 6),
    (4, 24),
    (5, 120),
    (10, 3628800),
])
def test_factorial_with_parametrized_inputs(input_number:int, expected_factorial:int) -> None:
    assert factorial(input_number) == expected_factorial

@pytest.mark.parametrize("input_number", [
    -1,
    -5,
    -10,
])
def test_factorial_with_negative_inputs(input_number:int) -> None:
    with pytest.raises(ValueError):
        factorial(input_number)


def test_factorial_with_non_integer_input() -> None:
    with pytest.raises(TypeError):
        factorial(2.5) # type: ignore