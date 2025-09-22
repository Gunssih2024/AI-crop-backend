import joblib
import pandas as pd
import numpy as np
import json

data = [
    [6.75, 3.0, 6.75, 28, 65, 6.0, True, False, False],
    [22.00975, 10.56468, 19.36858, 25, 80, 6.5, False, False, True],
    [18.69237, 8.30772, 11.76927, 22, 70, 6.0, False, True, False],
    [49.454, 23.73792, 43.51952, 25, 80, 6.5, False, False, True],
    [11.91675, 5.72004, 10.48674, 25, 80, 6.5, False, False, True],
]
data2 = [
    [
        53.925250000000005,
        25.884120000000003,
        47.45422,
        25,
        80,
        6.5,
        False,
        False,
        True,
    ]
]

columns = [
    "N_req_kg_per_ha",
    "P_req_kg_per_ha",
    "K_req_kg_per_ha",
    "Temperature_C",
    "Humidity_%",
    "pH",
    "Crop_cotton",
    "Crop_maize",
    "Crop_rice",
]

df = pd.DataFrame(data2, columns=columns)
loaded_preprocessor = joblib.load("preprocessor_joblib.pkl")
transformed_data = loaded_preprocessor.transform(df)

print(transformed_data)

loaded_model = joblib.load("./random_forest_model_joblib.pkl")

predictions = loaded_model.predict(transformed_data)
print(predictions)
