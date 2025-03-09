# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Functions for the auxiliary operations."""

import os
import pathlib
import random

import numpy as np
import torch

BOLD_UNDERLINE = "============================================"
THIN_UNDERLINE = "--------------------------------------------"


def _get_absolute_path() -> str:
    """Get absolute path of the library's sources."""
    return str(pathlib.Path(__file__).parent)


def fix_random_seed(seed: int) -> None:
    """Fix pseudo-random number generator seed for reproducible tests."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["RANDOM_SEED"] = str(seed)


NumericType = int | float | np.number
