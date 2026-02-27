# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ModelHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        self.sam_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_base_plus.pt"

        sam2_model = build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    # def handle(self, image):
    #     self.predictor.set_image(np.array(image))
    #     features = self.predictor.get_image_embedding()
    #     return features
    
    def handle(self, image):
        img = np.ascontiguousarray(np.asarray(image))
        img = img.copy()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(img)
            features = self.predictor.get_image_embedding()
        return features