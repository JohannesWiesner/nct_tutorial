"""Time-system conventions shared by configuration, model, and UI helpers."""

from __future__ import annotations

import numpy as np


# nctpy samples continuous control solutions at this fixed interval.
CONTINUOUS_TIME_STEP = 0.001


def validate_time_system(system: str) -> str:
    """Accept one of the two time systems supported by the tutorial."""
    if system not in ("discrete", "continuous"):
        raise ValueError("Time system must be 'discrete' or 'continuous'")
    return system


def validate_horizon(value: float, system: str = "discrete") -> int | float:
    """Read a positive horizon using the selected system's time units.

    Discrete T is an integer number of transitions. Continuous T is a
    duration in multiples of 0.001, matching nctpy's sample interval so
    the last plotted sample and the solver's target time coincide.
    """
    validate_time_system(system)
    message = ("T must be a positive integer for discrete time" if system == "discrete"
               else "T must be a positive duration in multiples of 0.001 for continuous time")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(number) or number <= 0:
        raise ValueError(message)
    if system == "discrete":
        if not number.is_integer():
            raise ValueError(message)
        return int(number)
    steps = number / CONTINUOUS_TIME_STEP
    if steps < 1 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError(message)
    return round(steps) * CONTINUOUS_TIME_STEP
