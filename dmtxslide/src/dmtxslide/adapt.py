"""Per-image adaptation — where source/printer/color diversity is absorbed.

Nothing here carries a printer-specific constant. Each function derives its
behaviour from a measured property of the image in hand, so an unseen label
colour or module size is handled at runtime rather than by retuning.

Polarity is intentionally NOT handled here: libdmtx already searches both
dark-on-light and light-on-dark internally (see region.polarity), so manual
inversion would only add cost.
"""
from __future__ import annotations

import cv2
import numpy as np


def _otsu_separability(gray: np.ndarray) -> float:
    """Between-class variance at the Otsu split — how bimodal the image is.
    A code separates cleanly from its substrate in *some* projection; we pick
    that projection rather than assuming luminance."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(np.nanmax(sigma_b))


def best_contrast_channel(image: np.ndarray) -> np.ndarray:
    """Project colour -> the single channel where the code is most separable.

    Black-on-yellow separates in R/G, not blue or luminance; pink, green and
    blue label stock each peak elsewhere. Picking by separability generalises
    across label colours without enumerating them.
    """
    if image.ndim == 2:
        return image
    b, g, r = cv2.split(image[:, :, :3])
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    candidates = {"gray": gray, "B": b, "G": g, "R": r}
    return max(candidates.values(), key=_otsu_separability)


def normalize_scale(gray: np.ndarray, module_px: float | None,
                    target_px: float = 8.0, max_dim: int = 1600) -> np.ndarray:
    """Resize so one module ~ target_px (libdmtx's comfortable band).

    Size-agnosticism comes from normalising the one variable libdmtx is most
    sensitive to, instead of hoping the capture arrives in range. No-op if we
    have no estimate yet.
    """
    if not module_px or module_px <= 0:
        return gray
    factor = target_px / module_px
    if abs(factor - 1.0) < 0.15:
        return gray
    h, w = gray.shape
    if max(h, w) * factor > max_dim:
        factor = max_dim / max(h, w)
    interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=interp)
