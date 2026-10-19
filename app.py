import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Manufacturing Equipment Output Predictor API")

# Load the trained model pipeline
model = joblib.load("manufacturing_output_model.pkl")

# Define the input schema using Pydantic
class ProductionParameters(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: Optional[float] = None
    Ambient_Temperature: Optional[float] = None
    Machine_Age: float
    Operator_Experience: Optional[float] = None
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float
    Timestamp_Hour: int
    Timestamp_Month: int

@app.get("/")
def home():
    return {"message": "Manufacturing Output Prediction API is online! Use the /predict endpoint for inference."}

@app.post("/predict")
def predict(data: ProductionParameters):
    # Convert incoming request data to a Pandas DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Generate prediction using the pipeline
    prediction = model.predict(input_df)[0]
    
    return {
        "status": "success",
        "predicted_parts_per_hour": round(prediction, 2)
    }
