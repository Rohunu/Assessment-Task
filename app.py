from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()
security = HTTPBasic()

# Load the model
model = load_model("image_classifier.h5")
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Basic authentication
def verify_credentials(credentials: HTTPBasicCredentials):
    correct_username = "admin"
    correct_password = "password123"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/predict")
async def predict(file: UploadFile = File(...), credentials: HTTPBasicCredentials = Depends(security)):
    verify_credentials(credentials)
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))