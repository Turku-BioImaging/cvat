# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import numpy as np
import torch

# needed this dependency for polygon extraction:
import cv2  

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class SAM2Result:
    mask_u8: np.ndarray          # (H, W) uint8 0/1
    polygon: List[float]         # [x1, y1, x2, y2, ...]
    score: float


class ModelHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2] device={self.device}")

        # Adjust to your actual paths inside the Nuclio container
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.sam_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_large.pt"

        sam2_model = build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    @staticmethod
    def _to_point_arrays(
        pos_points: Iterable[Iterable[float]],
        neg_points: Iterable[Iterable[float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pos = np.array(list(pos_points or []), dtype=np.float32)
        neg = np.array(list(neg_points or []), dtype=np.float32)

        if pos.size == 0 and neg.size == 0:
            return None, None

        coords = []
        labels = []

        if pos.size > 0:
            coords.append(pos)
            labels.append(np.ones((pos.shape[0],), dtype=np.int32))

        if neg.size > 0:
            coords.append(neg)
            labels.append(np.zeros((neg.shape[0],), dtype=np.int32))

        point_coords = np.concatenate(coords, axis=0).astype(np.float32)
        point_labels = np.concatenate(labels, axis=0).astype(np.int32)

        return point_coords, point_labels

    @staticmethod
    def _mask_to_polygon(mask_u8: np.ndarray) -> List[float]:
        """
        Convert a binary mask to a single polygon (outer contour).
        CVAT interactors typically return one polygon (CVAT polygons are single rings in this interactor flow).
        """

        # Find external contours only
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Choose the largest contour by area
        largest = max(contours, key=cv2.contourArea)
        if largest is None or len(largest) < 3:
            return []

        # Flatten to [x1, y1, x2, y2, ...]
        pts = largest.reshape(-1, 2).astype(np.float32)
        polygon = pts.ravel().tolist()

        return polygon

    def handle(
        self,
        image_rgb_uint8: np.ndarray,                 # (H, W, 3) uint8
        pos_points: list,
        neg_points: list,
        obj_bbox: Optional[list] = None,             # [x1,y1,x2,y2] or None
        threshold: float = 0.0,
        multimask_output: bool = True,
    ) -> Optional[SAM2Result]:
        """
        Returns one best mask + polygon suitable for CVAT interactor response.
        """

        point_coords, point_labels = self._to_point_arrays(pos_points, neg_points)

        box = None
        if obj_bbox is not None:
            box_arr = np.array(obj_bbox, dtype=np.float32).reshape(-1)
            if box_arr.size == 4:
                box = box_arr
            else:
                # Invalid bbox
                box = None
        
        # CVAT may send empty neg_points but still expects interactor to behave.
        # If we have no prompts at all, return None (caller returns empty response).
        if point_coords is None and box is None:
            return None


        use_cuda_amp = (self.device == "cuda")

        with torch.inference_mode():
            if use_cuda_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    self.predictor.set_image(image_rgb_uint8)
                    out = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=multimask_output,
                    )
            else:
                self.predictor.set_image(image_rgb_uint8)
                out = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output,
                )

        # SAM-style predictors often return (masks, scores, logits)
        if isinstance(out, tuple) and len(out) == 3:
            masks, scores, _ = out
        elif isinstance(out, tuple) and len(out) == 2:
            masks, scores = out
        else:
            raise RuntimeError(f"Unexpected predictor output: {type(out)}")

        # Normalize shapes:
        # masks: (K, H, W) or (H, W) depending on predict settings
        masks = np.asarray(masks)
        scores = np.asarray(scores).astype(np.float32)

        if masks.ndim == 2:
            masks = masks[None, :, :]

        if masks.shape[0] == 0:
            return None

        # Pick best by score
        best_idx = int(scores.argmax()) if scores.size else 0
        best_score = float(scores[best_idx]) if scores.size else 0.0

        if best_score < float(threshold):
            return None

        best_mask = masks[best_idx]

        # Convert to 0/1 uint8
        # Some predictors return bool; some return float 0..1
        best_mask_u8 = (best_mask > 0.0).astype(np.uint8)

        polygon = self._mask_to_polygon(best_mask_u8)

        return SAM2Result(mask_u8=best_mask_u8, polygon=polygon, score=best_score)
    
  