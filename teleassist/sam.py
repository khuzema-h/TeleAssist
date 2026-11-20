"""SAM 2 image segmentation for TeleAssist scans (no CoTracker)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_REPO_PARENT = Path(__file__).resolve().parents[2]
_DEFAULT_CKPT = _REPO_PARENT / "checkpoints" / "sam2_hiera_base_plus.pt"
SAM_CONFIG = "configs/sam2/sam2_hiera_b+.yaml"


def _checkpoint_path() -> Path:
    override = os.environ.get("TELEASSIST_SAM_CKPT")
    if override:
        return Path(override)
    return _DEFAULT_CKPT


class SamSegmenter:
    """Lazy SAM 2 predictor for single-region masks."""

    def __init__(self) -> None:
        path = _checkpoint_path()
        if not path.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {path}")
        self._checkpoint = path
        self.sam = None
        self._sam_model = None

    def _ensure_sam(self) -> None:
        if self.sam is not None:
            return
        import sam2  # noqa: F401
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._sam_model = build_sam2(SAM_CONFIG, str(self._checkpoint), device="cuda")
        self.sam = SAM2ImagePredictor(self._sam_model)

    def segment_region(
        self,
        rgb: np.ndarray,
        point: tuple[float, float] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        if (point is None) == (box is None):
            raise ValueError("Segmentation requires either a point or a box")
        image = np.ascontiguousarray(rgb[..., :3])
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._ensure_sam()
        try:
            self.sam.set_image(image)
            if point is not None:
                masks, scores, _ = self.sam.predict(
                    point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    multimask_output=True,
                )
            else:
                masks, scores, _ = self.sam.predict(
                    box=np.array(box, dtype=np.float32),
                    multimask_output=False,
                )
            mask = masks[int(np.argmax(scores))] > 0
            if not np.any(mask):
                raise RuntimeError("SAM 2 returned an empty mask")
            return mask
        finally:
            if self.sam is not None:
                self.sam.reset_predictor()

    def release_sam(self) -> None:
        self.sam = None
        self._sam_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return
