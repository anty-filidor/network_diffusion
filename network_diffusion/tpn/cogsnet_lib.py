"""Interface for C extended functions."""

from typing import List


def cogsnet(forgettingType: str, snapshotInterval: int, mu: float, theta: float, lambda_: float, units: int, pathEvents: str, delimiter: str) -> List[List[float]]:
    """Process a file and return a list of lists."""
    ...
