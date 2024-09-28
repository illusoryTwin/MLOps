from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn 
from pydantic import BaseModel

# Load the trained model
app = FastAPI()

model_path = "../models/co2_emission_model"


model = mlflow.sklearn.load_model(model_path)

class InputData(BaseModel):
    Model_Year: int 
    Make: str
    Model: str 
    Vehicle_Class: str 
    Engine_Size: float 
    Cylinders: int 
    Transmission: str 
    Fuel_Consumption_in_City: float
    Fuel_Consumption_in_City_Hwy: float
    Fuel_Consumption_comb: float 
    Smog_Level: int 

@app.post("/predict")
def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(input_df)

    return {"CO2_Emissions": prediction[0]}
    

if __name__== "__main__":
    import uvicorn 
    uvicorn.run(app, host="127.0.0.1", port=8000)