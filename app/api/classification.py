from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

import io
from PIL import Image

from app.models.classification_model import classification_prediction, classification_model, classification_transform , labels_mapping
from app.config import device

router = APIRouter()

@router.post("/classification")
async def predict_classification(file: UploadFile = File(...)):
    try:

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB") 

        result = classification_prediction(image, classification_model, classification_transform, device)

    
        return {
            "classification_label": labels_mapping[result]
        }   
    

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    