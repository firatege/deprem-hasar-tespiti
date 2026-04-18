import numpy as np
import pytest

from scripts.benchmark_scalar_models import _parse_model_types
from scripts.train_scalar_baseline import SUPPORTED_MODEL_TYPES, build_model


def test_build_model_all_supported_types_can_fit() -> None:
    x = np.array(
        [
            [0.1, 0.0, 1.0],
            [0.2, 0.1, 0.9],
            [0.0, 0.2, 0.8],
            [1.0, 0.9, 0.1],
            [0.9, 0.8, 0.0],
            [0.8, 1.0, 0.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    for model_type in SUPPORTED_MODEL_TYPES:
        model = build_model(model_type)
        model.fit(x, y)
        probs = model.predict_proba(x)
        assert probs.shape == (6, 2)


def test_parse_model_types_deduplicates_and_validates() -> None:
    parsed = _parse_model_types("logreg,random_forest,logreg")
    assert parsed == ["logreg", "random_forest"]

    with pytest.raises(ValueError):
        _parse_model_types("logreg,unknown_model")

