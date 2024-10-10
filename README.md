# CO2 Emission Prediction

This project incorporates a model that predicts CO2 emission levels based on various vehicle parameters. The model is deployed using **FastAPI**, and a web application is provided to interact with the API for making predictions.

### List of Parameters (example data for inference):
- `Model_Year`
- `Make`
- `Model`
- `Vehicle_Class`
- `Engine_Size`
- `Cylinders`
- `Transmission`
- `Fuel_Consumption_in_City (L/100 km)`
- `Fuel_Consumption_on_Highway (L/100 km)`
- `Fuel_Consumption_Combined (L/100 km)`
- `CO2_Emissions`
- `Smog_Level`

### Example Data:
`2021,Acura,ILX,Compact,2.4,4,AM8,9.9,7,8.6,199,3`

### Launch Instructions

To launch the model, use the following command to start the Docker container:

```bash
docker compose up --build
