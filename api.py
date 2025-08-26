from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# تحميل النموذج والمحول
model = joblib.load("mymodel1.pkl")
tf = joblib.load("tf1.pkl")

app = FastAPI()

# تفعيل CORS عشان تقدر تتصل بيه من Flutter Web أو الموبايل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # تقدر تغيّرها لدومين تطبيقك بس
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaultRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: FaultRequest):
    text = request.text.strip()
    if not text:
        return {"error": "من فضلك أدخل وصف العطل"}
    vec = tf.transform([text])
    prediction = model.predict(vec)[0]
    return {"prediction": prediction}
