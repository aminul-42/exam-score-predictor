from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import numpy as np
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8001", "http://localhost:5500", "http://localhost:3148", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class InputData(BaseModel):
    hours: float

# Define the LinearRegression model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One input feature (hours), one output (result)

    def forward(self, x):
        return self.linear(x)

# Load the trained model
try:
    model_path = "linearregression.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found in {os.getcwd()}")
    
    # Try loading as state_dict first
    try:
        model = LinearRegression()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded successfully as state_dict from {model_path}")
    except Exception as state_dict_error:
        print(f"Failed to load as state_dict: {state_dict_error}")
        # Try loading as full model
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            print(f"Model loaded successfully as full model from {model_path}")
        except Exception as full_model_error:
            raise Exception(f"Failed to load model as full model: {full_model_error}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise Exception(f"Failed to load model: {e}")

@app.get("/")
async def serve_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "Server is running"}

@app.post("/predict")
async def predict(data: InputData):
    try:
        # Validate input
        if data.hours < 0:
            raise HTTPException(status_code=400, detail="Hours of study cannot be negative")
        
        # Convert input to tensor
        hours = np.array([data.hours], dtype=np.float32)
        hours_tensor = torch.tensor(hours).reshape(1, -1)

        # Make prediction
        with torch.no_grad():
            prediction = model(hours_tensor).item()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")