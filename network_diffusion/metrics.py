"""Metrics used to assess the diffusion."""

import warnings
from itertools import accumulate

import numpy as np


def compute_gain(exposed_nb: int, seeds_nb: int, actors_nb: int) -> float:
    """
    Compute gain from simulation to reflect relative spreading coverage.

    :param exposed_nb: number of activated actors during simulation
    :param seeds_nb: size of the seed set
    :param actors_nb: total number of actors in the network
    :return: gain in range [0, 1]
    """
    if exposed_nb < seeds_nb:
        raise ValueError("exposed_nb cannot be < seeds_nb")
    if exposed_nb > actors_nb:
        raise ValueError("exposed_nb cannot be > actors_nb")
    if seeds_nb >= actors_nb:
        raise ValueError("seeds_nb cannot be >= actors_nb")
    max_available_gain = actors_nb - seeds_nb
    obtained_gain = exposed_nb - seeds_nb
    return 100 * obtained_gain / max_available_gain


def compute_area(
    expositions_rec: list[int], actors_nb: int, trim_tail: bool
) -> float | None:
    """
    Compute norm. AuC from expositions while seed set impact is discarded.

    :param expositions_rec: record of NEW activations in each simulation step
    :param actors_nb: number of actors in the network
    :param trim_tail: a flag indicating whether to consider last elements of
        the record which have the same value
    :return: area under the cumulated activations curve in range [0, 1]
    """
    # convert expositions into a cumulative sum
    cumsum = np.array(list(accumulate(expositions_rec)))

    # find index at which cut cumsum due to stable state reached
    if trim_tail:
        cumsum_diff = cumsum[1:] - cumsum[:-1]
        if sum(cumsum_diff) == 0:
            warnings.warn(
                "cum. dist. must contain at least two diff. samples.",
                stacklevel=1,
            )
            return None
        idx = 0
        for element in cumsum_diff[::-1]:
            if element != 0:
                break
            idx += 1
        cutoff = len(cumsum) - idx
        cumsum = cumsum[:cutoff]

    # check the final cumsum
    if len(cumsum) < 2:
        warnings.warn(
            "cum. dist. must contain at least two samples.",
            stacklevel=1,
        )
        return None
    if cumsum[-1] > actors_nb:
        raise ValueError("Number of activations cannot be > actors_nb")

    # if everything's ok compute the area
    seeds_nb = expositions_rec[0]
    cumsum_scaled = (cumsum - seeds_nb) / (actors_nb - seeds_nb)
    cumsum_steps = np.linspace(0, 1, len(cumsum_scaled))
    return np.trapezoid(cumsum_scaled, cumsum_steps)
