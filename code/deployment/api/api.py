from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn 
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from code.models.preprocessing import Preprocessor # type: ignore

# Load the trained model
app = FastAPI()

model_path = "code/models/co2_emission_model"


model = mlflow.sklearn.load_model(model_path)

preprocessor = Preprocessor()

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
def predict(input: InputData):
    input_df = pd.DataFrame([input.model_dump()])

    # Preprocess the input data
    processed_data = preprocessor.preprocess_input(input_df)
    prediction = model.predict(processed_data)
    
    return {"CO2_Emissions": prediction[0]}
    # return {"CO2_Emissions": prediction}


if __name__== "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # data = {
    #     "Model_Year":2021,
    #     "Make":"Acura",
    #     "Model":"ILX",
    #     "Vehicle_Class":"Compact",
    #     "Engine_Size":2.4,
    #     "Cylinders":4,
    #     "Transmission":"AM8",
    #     "Fuel_Consumption_in_City": 9.9,
    #     "Fuel_Consumption_in_City_Hwy": 7,
    #     "Fuel_Consumption_comb": 8.6,
    #     "Smog_Level": 3
    # }
    # print(predict(
    #     InputData(**data)))