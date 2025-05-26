import numpy as np
import pytest

from network_diffusion.metrics import compute_area, compute_gain


@pytest.mark.parametrize(
    "exposed_nb, seeds_nb, actors_nb, exp_gain",
    [
        (1, 1, 100, 0),
        (50, 10, 100, 44.444),
        (25, 5, 100, 21.053),
    ],
)
def test_compute_gain(exposed_nb, seeds_nb, actors_nb, exp_gain):
    np.testing.assert_almost_equal(
        compute_gain(
            exposed_nb=exposed_nb, seeds_nb=seeds_nb, actors_nb=actors_nb
        ),
        exp_gain,
        3,
    )


@pytest.mark.parametrize(
    "exposed_nb, seeds_nb, actors_nb",
    [
        (10, 15, 100),
        (150, 10, 100),
        (100, 100, 100),
    ],
)
def test_compute_gain_err(exposed_nb, seeds_nb, actors_nb):
    with pytest.raises(ValueError):
        compute_gain(
            exposed_nb=exposed_nb, seeds_nb=seeds_nb, actors_nb=actors_nb
        )


@pytest.mark.parametrize(
    "expositions_rec, actors_nb, trim_tail, exp_area",
    [
        ([2897, 33918, 20203, 742, 2, 0, 0], 57954, True, 0.773),
        ([10, 13, 24, 3], 100, True, 0.259),
        ([10, 13, 24, 3, 0], 100, True, 0.259),
        ([10, 13, 24, 3, 0, 0], 100, True, 0.259),
        ([10, 13, 24, 3, 0, 0, 0], 100, True, 0.259),
        ([2897, 33918, 20203, 742, 2, 0, 0], 57954, False, 0.848),
        ([10, 13, 24, 3], 100, False, 0.259),
        ([10, 13, 24, 3, 0], 100, False, 0.306),
        ([10, 13, 24, 3, 0, 0], 100, False, 0.333),
        ([10, 13, 24, 3, 0, 0, 0], 100, False, 0.352),
    ],
)
def test_compute_area(expositions_rec, actors_nb, trim_tail, exp_area):
    np.testing.assert_almost_equal(
        compute_area(
            expositions_rec=expositions_rec,
            actors_nb=actors_nb,
            trim_tail=trim_tail,
        ),
        exp_area,
        3,
    )


@pytest.mark.parametrize(
    "expositions_rec, actors_nb, trim_tail",
    [
        ([2897, 33918, 20203, 742, 2, 0, 0], 57761, True),
        ([2897, 33918, 20203, 742, 2, 0, 0], 57761, False),
    ],
)
def test_compute_area_err(expositions_rec, actors_nb, trim_tail):
    with pytest.raises(ValueError):
        compute_area(
            expositions_rec=expositions_rec,
            actors_nb=actors_nb,
            trim_tail=trim_tail,
        )


@pytest.mark.parametrize(
    "expositions_rec, actors_nb, trim_tail",
    [
        ([2897], 57954, False),
        ([2897, 0], 57954, True),
        ([2897, 0, 0], 57954, True),
    ],
)
def test_compute_area_none(expositions_rec, actors_nb, trim_tail):
    with pytest.warns(Warning):
        assert (
            compute_area(
                expositions_rec=expositions_rec,
                actors_nb=actors_nb,
                trim_tail=trim_tail,
            )
            is None
        )
