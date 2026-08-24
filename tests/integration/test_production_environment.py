from __future__ import annotations

import pytest

from pipeline_v4.src.run_workflow import validate_production_environment


def test_expected_production_environment_is_accepted():
    assert validate_production_environment(
        python_version=(3, 12), sklearn_version="1.5.2",
    ) == {"python": "3.12", "scikit_learn": "1.5.2", "status": "compatible"}


@pytest.mark.parametrize(
    ("python_version", "sklearn_version", "expected"),
    [
        ((3, 11), "1.5.2", "Python 3.12 is required"),
        ((3, 12), "1.6.0", "scikit-learn 1.5.2 is required"),
    ],
)
def test_incompatible_production_environment_fails_clearly(
    python_version, sklearn_version, expected,
):
    with pytest.raises(RuntimeError, match=expected):
        validate_production_environment(
            python_version=python_version, sklearn_version=sklearn_version,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"modal_classifier": "random_forest"}, "must be 'hybrid'"),
        ({"classifier_hash": "0" * 64}, "SHA256 does not match"),
    ],
)
def test_frozen_classifier_identity_fails_closed(kwargs, expected):
    with pytest.raises(RuntimeError, match=expected):
        validate_production_environment(
            python_version=(3, 12), sklearn_version="1.5.2", **kwargs,
        )
