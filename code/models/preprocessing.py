# code/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        self.label_encoder_make = LabelEncoder()
        self.label_encoder_model = LabelEncoder()
        self.label_encoder_vehicle_class = LabelEncoder()
        self.label_encoder_transmission = LabelEncoder()

    def num2cat_transform(self, data: pd.DataFrame):
        data['Make'] = self.label_encoder_make.fit_transform(data['Make'])
        data['Model'] = self.label_encoder_model.fit_transform(data['Model'])
        data['Vehicle_Class'] = self.label_encoder_vehicle_class.fit_transform(data['Vehicle_Class'])
        data['Transmission'] = self.label_encoder_transmission.fit_transform(data['Transmission'])
        return data

    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        input_data.rename(columns={
            'Fuel_Consumption_in_City(L/100 km)': 'Fuel_Consumption_in_City',
            'Fuel_Consumption_in_City_Hwy(L/100 km)': 'Fuel_Consumption_in_City_Hwy',
            'Fuel_Consumption_in_City_comb(L/100 km)': 'Fuel_Consumption_comb'
        }, inplace=True)

        input_data = input_data[['Make', 'Model', 'Vehicle_Class', 'Engine_Size',
                                 'Transmission', 'Fuel_Consumption_in_City', 'Smog_Level']]

        input_data = self.num2cat_transform(input_data)

        return input_data
