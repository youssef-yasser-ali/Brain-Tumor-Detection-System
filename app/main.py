from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.api import classification, segmentation

app = FastAPI()

templates = Jinja2Templates(directory="app")



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Routes

app.include_router(segmentation.router, prefix="/api", tags=["Segmentation"])
app.include_router(classification.router, prefix="/api", tags=["Classification"])
