import json
import pickle
from pathlib import Path

import numpy as np

from app.ml.scalar_baseline import build_scalar_features, compute_change_features


class DamageInferenceModel:
    def __init__(self, model_path: Path, config_path: Path) -> None:
        with model_path.open("rb") as fh:
            self._model = pickle.load(fh)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        self._threshold: float = float(cfg.get("threshold", 0.5))
        self._max_bands: int = int(cfg.get("max_bands", 3))
        self._image_only: bool = bool(cfg.get("image_only", False))
        self._normalize_pga: bool = bool(cfg.get("normalize_pga", False))

    def predict(
        self,
        *,
        pre_image: Path,
        post_image: Path,
        pga_value: float = 0.0,
        magnitude: float = 0.0,
        depth_km: float = 0.0,
        acquisition_delta_days: float = 0.0,
        building_density: float = 0.0,
    ) -> dict:
        features = compute_change_features(pre_image, post_image, max_bands=self._max_bands)
        if not self._image_only:
            features.extend(
                build_scalar_features(
                    pga_value=pga_value,
                    magnitude=magnitude,
                    depth_km=depth_km,
                    acquisition_delta_days=acquisition_delta_days,
                    building_density=building_density,
                    normalize_pga=self._normalize_pga,
                )
            )
        x = np.array([features], dtype=np.float32)
        if hasattr(self._model, "n_features_in_") and self._model.n_features_in_ != x.shape[1]:
            raise ValueError(
                f"Feature mismatch: model expects {self._model.n_features_in_}, got {x.shape[1]}"
            )
        prob = float(self._model.predict_proba(x)[0, 1])
        return {
            "probability": prob,
            "threshold": self._threshold,
            "binary_damage": int(prob >= self._threshold),
            "policy": "xview2_major_destroyed_is_one",
        }
