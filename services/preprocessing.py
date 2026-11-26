"""Configurable low-light filter: gamma correction and CLAHE.

The filter works on a single brightness channel (LAB lightness or HSV value)
so colors stay stable. Every step can be switched off, which lets
hyperparameter search compare filtered and unfiltered pipelines.

Not wired into the BentoML services yet; it is used by
``src/pipeline/tune_preprocessing.py`` to search for good parameters.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import cv2
import numpy as np

GammaMode = Literal["off", "fixed", "adaptive"]
Order = Literal["gamma_first", "clahe_first"]
Channel = Literal["lab", "hsv"]

# Adaptive gamma never brightens more strongly than this, to avoid
# amplifying sensor noise in near-black images.
MIN_ADAPTIVE_GAMMA = 0.35


@dataclass(frozen=True)
class FilterParams:
    gamma_mode: GammaMode = "off"
    gamma: float = 1.0                # used when gamma_mode == "fixed"
    target_lightness: float = 0.45    # used when gamma_mode == "adaptive"
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    order: Order = "gamma_first"
    channel: Channel = "lab"

    @property
    def is_identity(self) -> bool:
        return self.gamma_mode == "off" and not self.clahe

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mean_lightness(bgr: np.ndarray) -> float:
    """Mean grayscale value of a BGR image, normalized to 0-1."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0) if gray.size else 0.0


def _gamma_lut(gamma: float) -> np.ndarray:
    values = np.arange(256, dtype=np.float32) / 255.0
    return np.clip((values ** gamma) * 255.0, 0, 255).astype(np.uint8)


def _apply_gamma(channel: np.ndarray, params: FilterParams) -> np.ndarray:
    if params.gamma_mode == "fixed":
        gamma = params.gamma
    elif params.gamma_mode == "adaptive":
        mean = float(channel.mean()) / 255.0
        if not 0.0 < mean < params.target_lightness:
            return channel
        gamma = max(
            MIN_ADAPTIVE_GAMMA,
            math.log(params.target_lightness) / math.log(mean),
        )
    else:
        return channel
    return cv2.LUT(channel, _gamma_lut(gamma))


def _apply_clahe(channel: np.ndarray, params: FilterParams) -> np.ndarray:
    if not params.clahe:
        return channel
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip,
        tileGridSize=(params.clahe_tile, params.clahe_tile),
    )
    return clahe.apply(channel)


def apply_filter(bgr: np.ndarray, params: FilterParams) -> np.ndarray:
    """Return a filtered copy of a uint8 BGR image (the input is unchanged)."""
    if params.is_identity:
        return bgr

    if params.channel == "lab":
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        index, back = 0, cv2.COLOR_LAB2BGR
    else:
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        index, back = 2, cv2.COLOR_HSV2BGR

    channel = converted[:, :, index]
    if params.order == "gamma_first":
        channel = _apply_clahe(_apply_gamma(channel, params), params)
    else:
        channel = _apply_gamma(_apply_clahe(channel, params), params)

    converted[:, :, index] = channel
    return cv2.cvtColor(converted, back)
