from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image

from app.models.segmentation_model import segmentation_prediction, seg_model, seg_transformtion
from app.config import device

import numpy as np


router = APIRouter()

@router.post("/Segmentation")
async def predict_segmentaion(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB") 

        mask = segmentation_prediction(image, seg_model, seg_transformtion, device)
        mask_np = mask.numpy()
        

        mask_binary = (mask_np > 0.5).astype(np.uint8)

        return {"segmentation_mask": mask_binary.tolist()} 
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
