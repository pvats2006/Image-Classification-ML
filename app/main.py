from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.predictor import predict_image

UPLOAD_DIR = Path("temp")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Flower Classification API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = predict_image(str(file_path))

    file_path.unlink(missing_ok=True)

    return JSONResponse(result)