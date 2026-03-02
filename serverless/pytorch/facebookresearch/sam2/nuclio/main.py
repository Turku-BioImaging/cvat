# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT


import base64
import io
import json

import numpy as np
from PIL import Image

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("call handler")
    data = event.body

    # --- Parse CVAT interactor payload ---
    # CVAT sends pos_points, neg_points, and obj_bbox for interactors
    pos_points = data.get("pos_points", [])
    neg_points = data.get("neg_points", [])
    obj_bbox = data.get("obj_bbox", None)
    threshold = float(data.get("threshold", 0.0))

    buf = io.BytesIO(base64.b64decode(data["image"]))
    pil = Image.open(buf).convert("RGB")
    img = np.array(pil, dtype=np.uint8)

    result = context.user_data.model.handle(
        image_rgb_uint8=img,
        pos_points=pos_points,
        neg_points=neg_points,
        obj_bbox=obj_bbox,
        threshold=threshold,
        multimask_output=True,
    )

    return context.Response(
        body=json.dumps({
            "points": result.polygon,
            "mask": result.mask_u8.tolist(),
            "score": result.score,
        }),
        headers={},
        content_type="application/json",
        status_code=200,
    )








    